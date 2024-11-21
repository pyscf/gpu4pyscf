# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import ctypes
import cupy
import numpy as np
from gpu4pyscf.scf.int4c2e import _make_s_index_offsets, libgint
from gpu4pyscf.lib.cupy_helper import load_library, take_last2d
from gpu4pyscf.df.int3c2e import VHFOpt, make_fake_mol

def get_int2c2e_sorted(mol, intopt=None, direct_scf_tol=1e-13, aosym=None, omega=None, stream=None):
    '''
    Generated int2c2e consistent with pyscf
    '''
    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    if intopt is None:
        # Reuse int3c2e
        intopt = VHFOpt(mol, mol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=True)
    nao = mol.nao
    rows, cols = np.tril_indices(nao)

    nao_cart = intopt._sorted_mol.nao
    norb_cart = nao_cart + 1

    int2c = cupy.zeros([nao_cart, nao_cart], order='F')
    ao_offsets = np.array([nao_cart+1, nao_cart, nao_cart+1, nao_cart], dtype=np.int32)
    strides = np.array([1, nao_cart, nao_cart, nao_cart*nao_cart], dtype=np.int32)
    log_cutoff = np.log(direct_scf_tol)
    for k_id, log_q_k in enumerate(intopt.aux_log_qs):
        #bins_locs_k = _make_s_index_offsets(log_q_k, nbins)
        bins_locs_k = np.array([0, len(log_q_k)], dtype=np.int32)
        bins_floor_k = np.array([100], dtype=np.double)
        cp_k_id = k_id + len(intopt.log_qs)
        for l_id, log_q_l in enumerate(intopt.aux_log_qs):
            if k_id > l_id: continue
            #bins_locs_l = _make_s_index_offsets(log_q_l, nbins)
            bins_locs_l = np.array([0, len(log_q_l)], dtype=np.int32)
            bins_floor_l = np.array([100], dtype=np.double)
            cp_l_id = l_id + len(intopt.log_qs)
            nbins_locs_k = len(bins_locs_k) - 1
            nbins_locs_l = len(bins_locs_l) - 1
            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int2c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_k.ctypes.data_as(ctypes.c_void_p),
                bins_locs_l.ctypes.data_as(ctypes.c_void_p),
                bins_floor_k.ctypes.data_as(ctypes.c_void_p),
                bins_floor_l.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins_locs_k),
                ctypes.c_int(nbins_locs_l),
                ctypes.c_int(cp_k_id),
                ctypes.c_int(cp_l_id),
                ctypes.c_double(log_cutoff),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError("int2c2e failed\n")

    int2c[rows, cols] = int2c[cols, rows]
    if not mol.cart:
        coeff = intopt.cart2sph
        int2c = coeff.T @ int2c @ coeff

    return int2c

def get_int2c2e_ip_sorted(mol, auxmol, intopt=None, direct_scf_tol=1e-13, intor=None, aosym=None, stream=None):
    '''
    TODO: WIP
    '''
    if stream is None: stream = cupy.cuda.get_current_stream()
    if intopt is None:
        intopt = VHFOpt(mol, auxmol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=False)

    nbins = 1

    nao_cart = intopt.mol.nao
    naux_cart = intopt.auxmol.nao
    norb_cart = nao_cart + naux_cart + 1
    rows, cols = np.tril_indices(naux_cart)

    int2c = cupy.zeros([naux_cart, naux_cart], order='F')
    ao_offsets = np.array([nao_cart+1, nao_cart, nao_cart+1, nao_cart], dtype=np.int32)
    strides = np.array([1, naux_cart, naux_cart, naux_cart*naux_cart], dtype=np.int32)
    for k_id, log_q_k in enumerate(intopt.aux_log_qs):
        bins_locs_k = _make_s_index_offsets(log_q_k, nbins)
        cp_k_id = k_id + len(intopt.log_qs)
        for l_id, log_q_l in enumerate(intopt.aux_log_qs):
            if k_id > l_id: continue
            bins_locs_l = _make_s_index_offsets(log_q_l, nbins)
            cp_l_id = l_id + len(intopt.log_qs)
            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int2c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_k.ctypes.data_as(ctypes.c_void_p),
                bins_locs_l.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_k_id),
                ctypes.c_int(cp_l_id))

            if err != 0:
                raise RuntimeError("int2c2e failed\n")

    int2c[rows, cols] = int2c[cols, rows]
    if not auxmol.cart:
        coeff = intopt.aux_cart2sph
        int2c = coeff.T @ int2c @ coeff

    return int2c

def get_int2c2e(mol, direct_scf_tol=1e-13):
    '''
    Generate int2c2e on GPU
    '''
    intopt = VHFOpt(mol, mol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=True)
    int2c = get_int2c2e_sorted(mol, intopt=intopt)
    int2c = intopt.unsort_orbitals(int2c, axis=[0,1])
    return int2c
