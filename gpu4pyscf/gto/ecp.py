# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import load_library, contract
from gpu4pyscf.gto.mole import group_basis

libecp = load_library('libgecp')

ecp_cart_argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

libecp.ECP_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ip_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ipipv_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ipvip_cart.argtypes = ecp_cart_argtypes

ECP_ATOM_ID = 7

def sort_ecp_basis(_ecpbas, cart=True, log=None):
    '''
    # Sort ECP basis based on angular momentum
    # Remove SO Type basis functions
    '''
    not_so_type = _ecpbas[:, gto.SO_TYPE_OF] == 0
    _ecpbas = _ecpbas[not_so_type]

    # Sort ECP basis based on angular momentum and atom_id
    l_atm = _ecpbas[:,[gto.ANG_OF, gto.ATOM_OF]]

    uniq_l_atm, inv_idx, l_atm_counts = np.unique(
        l_atm, return_inverse=True, return_counts=True, axis=0)
    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    _ecpbas = _ecpbas[sorted_idx]

    # Group ECP basis based on angular momentum and atom id
    # Each group contains basis with multiple power order
    ecp_loc = np.append(0, np.cumsum(l_atm_counts))

    # Further group based on angular momentum for counting
    uniq_l, l_counts = np.unique(uniq_l_atm[:,0], return_counts=True, axis=0)

    return _ecpbas, uniq_l, l_counts, ecp_loc

def make_tasks(l_ctr_offsets, lecp_ctr_offsets):
    tasks = {}
    n_groups = len(l_ctr_offsets) - 1
    n_ecp_groups = len(lecp_ctr_offsets) - 1

    # TODO: Add screening here
    for i in range(n_groups):
        for j in range(i,n_groups):
            for k in range(n_ecp_groups):
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                ksh0, ksh1 = lecp_ctr_offsets[k], lecp_ctr_offsets[k+1]
                grid = np.meshgrid(
                    np.arange(ish0,ish1),
                    np.arange(jsh0,jsh1),
                    np.arange(ksh0,ksh1))
                grid = np.stack(grid, axis=-1).reshape(-1, 3)
                idx = grid[:,0] <= grid[:,1]
                tasks[i,j,k] = grid[idx]
    return tasks

def make_full_tasks(l_ctr_offsets, lecp_ctr_offsets):
    tasks = {}
    n_groups = len(l_ctr_offsets) - 1
    n_ecp_groups = len(lecp_ctr_offsets) - 1

    # TODO: Add screening
    for i in range(n_groups):
        for j in range(n_groups):
            for k in range(n_ecp_groups):
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                ksh0, ksh1 = lecp_ctr_offsets[k], lecp_ctr_offsets[k+1]
                grid = np.meshgrid(
                    np.arange(ish0,ish1),
                    np.arange(jsh0,jsh1),
                    np.arange(ksh0,ksh1))
                grid = np.stack(grid, axis=-1).reshape(-1, 3)
                tasks[i,j,k] = grid
    return tasks

def select_basis(ecpbas, ecp_atoms):
    """
    Select ECP basis for given ECP atoms, and reindexing atoms
    """
    atom_map = {}
    for idx, ecp_atom in enumerate(ecp_atoms):
        atom_map[ecp_atom] = idx

    selected_ecpbas = []
    for idx, bas in enumerate(ecpbas):
        atm_id = bas[gto.ATOM_OF]
        if atm_id in ecp_atoms:
            bas_copy = bas.copy()
            bas_copy[ECP_ATOM_ID] = atom_map[atm_id]
            selected_ecpbas.append(bas_copy)
    return np.array(selected_ecpbas)

def get_ecp(mol):
    """
    Calculate sum of ECP integrals

    Returns:
        CuPy array: [nao, nao]
            sum of ECP integrals over all ecp atoms
    """
    _sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol)

    _ecpbas = _sorted_mol._ecpbas
    _ecpbas, uniq_lecp, lecp_counts, ecp_loc= sort_ecp_basis(_ecpbas)

    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))

    tasks_all = make_tasks(l_ctr_offsets, lecp_offsets)

    atm = cp.asarray(_sorted_mol._atm, dtype=np.int32)
    bas = cp.asarray(_sorted_mol._bas, dtype=np.int32)
    env = cp.asarray(_sorted_mol._env, dtype=np.float64)

    ecpbas = cp.asarray(_ecpbas, dtype=np.int32)
    ecploc = cp.asarray(ecp_loc, dtype=np.int32)
    n_groups = len(uniq_l_ctr)
    n_ecp_groups = len(uniq_lecp)
    ao_loc = _sorted_mol.ao_loc_nr(cart=True)
    nao = ao_loc[-1]
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)

    mat1 = cp.zeros([nao, nao])
    for i in range(n_groups):
        for j in range(i,n_groups):
            for k in range(n_ecp_groups):
                tasks = cp.asarray(tasks_all[i,j,k], dtype=np.int32, order='F')
                ntasks = len(tasks)
                li = uniq_l_ctr[i,0]
                lj = uniq_l_ctr[j,0]
                lk = uniq_lecp[k]
                err = libecp.ECP_cart(
                    mat1.data.ptr, ao_loc.data.ptr, nao,
                    tasks.data.ptr, ntasks,
                    ecpbas.data.ptr, ecploc.data.ptr,
                    atm.data.ptr, bas.data.ptr, env.data.ptr,
                    li, lj, lk)
                if err != 0:
                    raise RuntimeError('ECP CUDA kernel failed.')
    coeff = cp.asarray(coeff)
    return coeff.T @ mat1 @ coeff


def get_ecp_ip(mol, ip_type='ip', ecp_atoms=None):
    """
    First derivative of ECP integrals

    Returns:
        CuPy array: [n_ecp_atoms, 3, nao, nao],
            reindex the first dimension acoording to ecp_atoms
    """
    if ecp_atoms is None:
        ecp_atoms = sorted(set(mol._ecpbas[:,gto.ATOM_OF]))

    if ip_type == 'ip':
        fn = libecp.ECP_ip_cart
        comp = 3
    else:
        raise ValueError('Invalid IP type')

    _sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol)
    _ecpbas = mol._ecpbas.copy()

    ecpbas = select_basis(_ecpbas, ecp_atoms)
    ecpbas, uniq_lecp, lecp_counts, ecp_loc= sort_ecp_basis(ecpbas)

    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))

    tasks_all = make_full_tasks(l_ctr_offsets, lecp_offsets)

    atm = cp.asarray(_sorted_mol._atm, dtype=np.int32)
    bas = cp.asarray(_sorted_mol._bas, dtype=np.int32)
    env = cp.asarray(_sorted_mol._env, dtype=np.float64)

    ecpbas = cp.asarray(ecpbas, dtype=np.int32)
    ecploc = cp.asarray(ecp_loc, dtype=np.int32)
    n_groups = len(uniq_l_ctr)
    n_ecp_groups = len(uniq_lecp)
    ao_loc = _sorted_mol.ao_loc_nr(cart=True)
    nao = ao_loc[-1]
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    n_ecp_atm = len(ecp_atoms)

    mat1 = cp.zeros([n_ecp_atm, comp, nao, nao])
    for i in range(n_groups):
        for j in range(n_groups):
            for k in range(n_ecp_groups):
                tasks = cp.asarray(tasks_all[i,j,k], dtype=np.int32, order='F')
                ntasks = len(tasks)
                li = uniq_l_ctr[i,0]
                lj = uniq_l_ctr[j,0]
                lk = uniq_lecp[k]
                err = fn(
                    mat1.data.ptr, ao_loc.data.ptr, nao,
                    tasks.data.ptr, ntasks,
                    ecpbas.data.ptr, ecploc.data.ptr,
                    atm.data.ptr, bas.data.ptr, env.data.ptr,
                    li, lj, lk)
                if err != 0:
                    raise RuntimeError('ECP CUDA kernel failed.')

    coeff = cp.asarray(coeff)
    mat1 = contract('axij,jq->axiq', mat1, coeff)
    mat1 = contract('axiq,ip->axpq', mat1, coeff)
    return mat1

def get_ecp_ipip(mol, ip_type='ipipv', ecp_atoms=None):
    """
    Second derivatives of ECP integrals
    Args:
        ip_type:
            ipipv -> (i''|ecp|j)
            ipvip -> (i'|ecp|j')

    Returns:
        CuPy array: [n_ecp_atoms, 9, nao, nao],
            reindex the first dimension acoording to ecp_atoms
    """
    if ecp_atoms is None:
        ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])

    if ip_type == 'ipipv':
        fn = libecp.ECP_ipipv_cart
        comp = 9
    elif ip_type == 'ipvip':
        fn = libecp.ECP_ipvip_cart
        comp = 9
    else:
        raise ValueError('Invalid IP type')

    _sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol)

    _ecpbas = _sorted_mol._ecpbas
    ecpbas = select_basis(_ecpbas, ecp_atoms)
    ecpbas, uniq_lecp, lecp_counts, ecp_loc= sort_ecp_basis(ecpbas)

    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))

    tasks_all = make_full_tasks(l_ctr_offsets, lecp_offsets)

    atm = cp.asarray(_sorted_mol._atm, dtype=np.int32)
    bas = cp.asarray(_sorted_mol._bas, dtype=np.int32)
    env = cp.asarray(_sorted_mol._env, dtype=np.float64)

    ecpbas = cp.asarray(ecpbas, dtype=np.int32)
    ecploc = cp.asarray(ecp_loc, dtype=np.int32)
    n_groups = len(uniq_l_ctr)
    n_ecp_groups = len(uniq_lecp)
    ao_loc = _sorted_mol.ao_loc_nr(cart=True)
    nao = ao_loc[-1]
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    n_ecp_atm = len(ecp_atoms)

    mat1 = cp.zeros([n_ecp_atm, comp, nao, nao])
    for i in range(n_groups):
        for j in range(n_groups):
            for k in range(n_ecp_groups):
                tasks = cp.asarray(tasks_all[i,j,k], dtype=np.int32, order='F')
                ntasks = len(tasks)
                li = uniq_l_ctr[i,0]
                lj = uniq_l_ctr[j,0]
                lk = uniq_lecp[k]
                err = fn(
                    mat1.data.ptr, ao_loc.data.ptr, nao,
                    tasks.data.ptr, ntasks,
                    ecpbas.data.ptr, ecploc.data.ptr,
                    atm.data.ptr, bas.data.ptr, env.data.ptr,
                    li, lj, lk)
                if err != 0:
                    raise RuntimeError('ECP CUDA kernel failed.')

    coeff = cp.asarray(coeff)
    mat1 = contract('axij,jq->axiq', mat1, coeff)
    mat1 = contract('axiq,ip->axpq', mat1, coeff)
    return mat1
