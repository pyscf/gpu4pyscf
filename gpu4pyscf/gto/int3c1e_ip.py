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
import cupy as cp
import numpy as np

from pyscf.lib import c_null_ptr
from gpu4pyscf.lib.cupy_helper import load_library, cart2sph, get_avail_mem


libgint = load_library('libgint')


def get_int3c1e_ip(mol, grids, charge_exponents, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    nao = mol.nao
    ngrids = grids.shape[0]
    total_double_number = ngrids * nao * nao * 6
    cp.get_default_memory_pool().free_all_blocks()
    avail_mem = get_avail_mem()
    reserved_available_memory = avail_mem // 4 # Leave space for further allocations
    allowed_double_number = reserved_available_memory // 8
    n_grid_split = int(np.ceil(total_double_number / allowed_double_number))
    if (n_grid_split > 100):
        raise Exception(f"Available GPU memory ({avail_mem / 1e9 : .1f} GB) is too small for "
                        "the 3 center integral first derivative, "
                        "which requires {total_double_number * 8 / 1e9 : .1f} GB of memory")
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    int3cip1_pinned_memory_pool = cp.cuda.alloc_pinned_memory(ngrids * nao * nao * 3 * np.array([1.0]).nbytes)
    int3c_ip1 = np.frombuffer(int3cip1_pinned_memory_pool, np.float64, ngrids * nao * nao * 3).reshape([3, ngrids, nao, nao], order='C')
    int3cip2_pinned_memory_pool = cp.cuda.alloc_pinned_memory(ngrids * nao * nao * 3 * np.array([1.0]).nbytes)
    int3c_ip2 = np.frombuffer(int3cip2_pinned_memory_pool, np.float64, ngrids * nao * nao * 3).reshape([3, ngrids, nao, nao], order='C')

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    for i_grid_split in range(0, ngrids, ngrids_per_split):
        ngrids_of_split = np.min([ngrids_per_split, ngrids - i_grid_split])
        int3c_grid_slice = cp.zeros([6, ngrids_of_split, nao, nao], order='C')
        for cp_ij_id, _ in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            stream = cp.cuda.get_current_stream()
            nao_cart = intopt._sorted_mol.nao

            log_q_ij = intopt.log_qs[cp_ij_id]

            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            ni = i1 - i0
            nj = j1 - j0

            ao_offsets = np.array([i0, j0], dtype=np.int32)
            strides = np.array([ni, ni*nj], dtype=np.int32)

            int3c_angular_slice = cp.zeros([6, ngrids_of_split, j1-j0, i1-i0], order='C')

            charge_exponents_pointer = c_null_ptr()
            if charge_exponents is not None:
                charge_exponents_pointer = charge_exponents[i_grid_split : i_grid_split + ngrids_of_split].data.ptr

            err = libgint.GINTfill_int3c1e_ip(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(grids[i_grid_split : i_grid_split + ngrids_of_split, :].data.ptr, ctypes.c_void_p),
                ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
                ctypes.c_int(ngrids_of_split),
                ctypes.cast(int3c_angular_slice.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError('GINTfill_int3c1e failed')

            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
            if not mol.cart:
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=2, ang=lj)
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=3, ang=li)

            int3c_grid_slice[:, :, j0:j1, i0:i1] = int3c_angular_slice

        ao_idx = np.argsort(intopt._ao_idx)
        grid_idx = np.arange(ngrids_of_split)
        derivative_idx = np.arange(6)
        int3c_grid_slice = int3c_grid_slice[np.ix_(derivative_idx, grid_idx, ao_idx, ao_idx)]

        # Each piece of the following memory is contiguous
        int3c_grid_slice[0, :, :, :].get(out = int3c_ip1[0, i_grid_split : i_grid_split + ngrids_of_split, :, :])
        int3c_grid_slice[1, :, :, :].get(out = int3c_ip1[1, i_grid_split : i_grid_split + ngrids_of_split, :, :])
        int3c_grid_slice[2, :, :, :].get(out = int3c_ip1[2, i_grid_split : i_grid_split + ngrids_of_split, :, :])
        int3c_grid_slice[3, :, :, :].get(out = int3c_ip2[0, i_grid_split : i_grid_split + ngrids_of_split, :, :])
        int3c_grid_slice[4, :, :, :].get(out = int3c_ip2[1, i_grid_split : i_grid_split + ngrids_of_split, :, :])
        int3c_grid_slice[5, :, :, :].get(out = int3c_ip2[2, i_grid_split : i_grid_split + ngrids_of_split, :, :])

    return int3c_ip1, int3c_ip2
