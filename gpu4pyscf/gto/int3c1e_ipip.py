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
import cupy as cp
import numpy as np
from pyscf import lib
from pyscf.gto import ATOM_OF
from pyscf.lib import c_null_ptr
from gpu4pyscf.lib.cupy_helper import load_library, cart2sph, get_avail_mem
from gpu4pyscf.gto.int3c1e import VHFOpt

libgint = load_library('libgint')

def get_int3c1e_ipip1_charge_contracted(mol, grids, charge_exponents, charges, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]
    charges = cp.asarray(charges).astype(np.float64)

    charges = charges.reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)

    int1e_charge_contracted = cp.empty([3, 3, mol.nao, mol.nao], order='C')
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        stream = cp.cuda.get_current_stream()

        log_q_ij = intopt.log_qs[cp_ij_id]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
        ni = i1 - i0
        nj = j1 - j0

        ao_offsets = np.array([i0, j0], dtype=np.int32)
        strides = np.array([ni, ni*nj], dtype=np.int32)

        charge_exponents_pointer = c_null_ptr()
        if charge_exponents is not None:
            charge_exponents_pointer = charge_exponents.data.ptr

        ngrids = grids.shape[0]
        # n_charge_sum_per_thread = 1 # means every thread processes one pair and one grid
        # n_charge_sum_per_thread = ngrids # or larger number gaurantees one thread processes one pair and all grid points
        n_charge_sum_per_thread = 100 # This number roughly optimize kernel performance on a large system

        int1e_angular_slice = cp.zeros([3, 3, j1-j0, i1-i0], order='C')

        err = libgint.GINTfill_int3c1e_ipip1_charge_contracted(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(grids.data.ptr, ctypes.c_void_p),
            ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.cast(int1e_angular_slice.data.ptr, ctypes.c_void_p),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.c_double(omega),
            ctypes.c_int(n_charge_sum_per_thread))

        if err != 0:
            raise RuntimeError('GINTfill_int3c1e_charge_contracted failed')

        int1e_angular_slice[1,0] = int1e_angular_slice[0,1]
        int1e_angular_slice[2,0] = int1e_angular_slice[0,2]
        int1e_angular_slice[2,1] = int1e_angular_slice[1,2]

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        if not mol.cart:
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=2, ang=lj)
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=3, ang=li)

        int1e_charge_contracted[:, :, i0:i1, j0:j1] = int1e_angular_slice.transpose(0,1,3,2)

    return intopt.unsort_orbitals(int1e_charge_contracted, axis=[2,3])

def get_int3c1e_ipvip1_charge_contracted(mol, grids, charge_exponents, charges, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]
    charges = cp.asarray(charges).astype(np.float64)

    charges = charges.reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)

    int1e_charge_contracted = cp.empty([3, 3, mol.nao, mol.nao], order='C')
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        stream = cp.cuda.get_current_stream()

        log_q_ij = intopt.log_qs[cp_ij_id]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
        ni = i1 - i0
        nj = j1 - j0

        ao_offsets = np.array([i0, j0], dtype=np.int32)
        strides = np.array([ni, ni*nj], dtype=np.int32)

        charge_exponents_pointer = c_null_ptr()
        if charge_exponents is not None:
            charge_exponents_pointer = charge_exponents.data.ptr

        ngrids = grids.shape[0]
        # n_charge_sum_per_thread = 1 # means every thread processes one pair and one grid
        # n_charge_sum_per_thread = ngrids # or larger number gaurantees one thread processes one pair and all grid points
        n_charge_sum_per_thread = 100 # This number roughly optimize kernel performance on a large system

        int1e_angular_slice = cp.zeros([3, 3, j1-j0, i1-i0], order='C')

        err = libgint.GINTfill_int3c1e_ipvip1_charge_contracted(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(grids.data.ptr, ctypes.c_void_p),
            ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.cast(int1e_angular_slice.data.ptr, ctypes.c_void_p),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.c_double(omega),
            ctypes.c_int(n_charge_sum_per_thread))

        if err != 0:
            raise RuntimeError('GINTfill_int3c1e_charge_contracted failed')

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        if not mol.cart:
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=2, ang=lj)
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=3, ang=li)

        int1e_charge_contracted[:, :, i0:i1, j0:j1] = int1e_angular_slice.transpose(0,1,3,2)

    return intopt.unsort_orbitals(int1e_charge_contracted, axis=[2,3])

def get_int3c1e_ip1ip2_charge_contracted(mol, grids, charge_exponents, charges, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]
    charges = cp.asarray(charges).astype(np.float64)

    charges = charges.reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)

    int1e_charge_contracted = cp.empty([3, 3, mol.nao, mol.nao], order='C')
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        stream = cp.cuda.get_current_stream()

        log_q_ij = intopt.log_qs[cp_ij_id]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
        ni = i1 - i0
        nj = j1 - j0

        ao_offsets = np.array([i0, j0], dtype=np.int32)
        strides = np.array([ni, ni*nj], dtype=np.int32)

        charge_exponents_pointer = c_null_ptr()
        if charge_exponents is not None:
            charge_exponents_pointer = charge_exponents.data.ptr

        ngrids = grids.shape[0]
        # n_charge_sum_per_thread = 1 # means every thread processes one pair and one grid
        # n_charge_sum_per_thread = ngrids # or larger number gaurantees one thread processes one pair and all grid points
        n_charge_sum_per_thread = 100 # This number roughly optimize kernel performance on a large system

        int1e_angular_slice = cp.zeros([3, 3, j1-j0, i1-i0], order='C')

        err = libgint.GINTfill_int3c1e_ip1ip2_charge_contracted(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(grids.data.ptr, ctypes.c_void_p),
            ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.cast(int1e_angular_slice.data.ptr, ctypes.c_void_p),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.c_double(omega),
            ctypes.c_int(n_charge_sum_per_thread))

        if err != 0:
            raise RuntimeError('GINTfill_int3c1e_charge_contracted failed')

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        if not mol.cart:
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=2, ang=lj)
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=3, ang=li)

        int1e_charge_contracted[:, :, i0:i1, j0:j1] = int1e_angular_slice.transpose(0,1,3,2)

    return intopt.unsort_orbitals(int1e_charge_contracted, axis=[2,3])

def get_int3c1e_ipip2_density_contracted(mol, grids, charge_exponents, dm, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    nao_cart = intopt._sorted_mol.nao
    ngrids = grids.shape[0]

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    dm = cp.asarray(dm)
    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1] and dm.shape[0] == mol.nao

    dm = intopt.sort_orbitals(dm, [0,1])
    if not mol.cart:
        cart2sph_transformation_matrix = intopt.cart2sph
        # TODO: This part is inefficient (O(N^3)), should be changed to the O(N^2) algorithm
        dm = cart2sph_transformation_matrix @ dm @ cart2sph_transformation_matrix.T
    dm = dm.flatten(order='F') # Column major order matches (i + j * n_ao) access pattern in the C function

    dm = cp.asnumpy(dm)

    ao_loc_sorted_order = intopt._sorted_mol.ao_loc_nr(cart = True)
    l_ij = intopt.l_ij.T.flatten()
    bas_coords = intopt._sorted_mol.atom_coords()[intopt._sorted_mol._bas[:, ATOM_OF]].flatten()

    n_total_hermite_density = intopt.density_offset[-1]
    dm_pair_ordered = np.empty(n_total_hermite_density)
    libgint.GINTinit_J_density_rys_preprocess(dm.ctypes.data_as(ctypes.c_void_p),
                                              dm_pair_ordered.ctypes.data_as(ctypes.c_void_p),
                                              ctypes.c_int(1), ctypes.c_int(nao_cart),
                                              ctypes.c_int(len(intopt.bas_pairs_locs) - 1),
                                              intopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                                              intopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                                              l_ij.ctypes.data_as(ctypes.c_void_p),
                                              intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
                                              ao_loc_sorted_order.ctypes.data_as(ctypes.c_void_p),
                                              bas_coords.ctypes.data_as(ctypes.c_void_p),
                                              ctypes.c_bool(False))

    dm_pair_ordered = cp.asarray(dm_pair_ordered)

    n_threads_per_block_1d = 16
    n_max_blocks_per_grid_1d = 65535
    n_max_threads_1d = n_threads_per_block_1d * n_max_blocks_per_grid_1d
    n_grid_split = int(np.ceil(ngrids / n_max_threads_1d))
    if (n_grid_split > 100):
        print(f"Grid dimension = {ngrids} is too large, more than 100 kernels for one electron integral will be launched.")
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    int3c_density_contracted = cp.zeros([3, 3, ngrids], order='C')

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        for cp_ij_id, _ in enumerate(intopt.log_qs):
            stream = cp.cuda.get_current_stream()

            log_q_ij = intopt.log_qs[cp_ij_id]

            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

            charge_exponents_pointer = c_null_ptr()
            if charge_exponents is not None:
                exponents_slice = charge_exponents[p0:p1]
                charge_exponents_pointer = exponents_slice.data.ptr
            grids_slice = grids[p0:p1]

            # n_pair_sum_per_thread = 1 # means every thread processes one pair and one grid
            # n_pair_sum_per_thread = nao_cart # or larger number gaurantees one thread processes one grid and all pairs of the same type
            n_pair_sum_per_thread = nao_cart

            err = libgint.GINTfill_int3c1e_ipip2_density_contracted(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(grids_slice.data.ptr, ctypes.c_void_p),
                ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
                ctypes.c_int(p1-p0),
                ctypes.cast(dm_pair_ordered.data.ptr, ctypes.c_void_p),
                intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
                ctypes.cast(int3c_density_contracted[:, p0:p1].data.ptr, ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(omega),
                ctypes.c_int(n_pair_sum_per_thread))

            if err != 0:
                raise RuntimeError('GINTfill_int3c1e_density_contracted failed')

    int3c_density_contracted[1,0] = int3c_density_contracted[0,1]
    int3c_density_contracted[2,0] = int3c_density_contracted[0,2]
    int3c_density_contracted[2,1] = int3c_density_contracted[1,2]

    return int3c_density_contracted

def int1e_grids_ipip1(mol, grids, charge_exponents=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    assert grids is not None
    assert charges is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    return get_int3c1e_ipip1_charge_contracted(mol, grids, charge_exponents, charges, intopt)

def int1e_grids_ipvip1(mol, grids, charge_exponents=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    assert grids is not None
    assert charges is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    return get_int3c1e_ipvip1_charge_contracted(mol, grids, charge_exponents, charges, intopt)

def int1e_grids_ip1ip2(mol, grids, charge_exponents=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    assert grids is not None
    assert charges is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    return get_int3c1e_ip1ip2_charge_contracted(mol, grids, charge_exponents, charges, intopt)

def int1e_grids_ipip2(mol, grids, charge_exponents=None, dm=None, direct_scf_tol=1e-13, intopt=None):
    assert grids is not None
    assert dm is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    return get_int3c1e_ipip2_density_contracted(mol, grids, charge_exponents, dm, intopt)
