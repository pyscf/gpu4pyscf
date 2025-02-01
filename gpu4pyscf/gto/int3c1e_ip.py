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
    
    buf_size = ngrids * nao * nao * 3
    int3cip1_pinned_buf = cp.cuda.alloc_pinned_memory(buf_size * 8)
    int3c_ip1 = np.frombuffer(int3cip1_pinned_buf, np.float64, buf_size).reshape([3, ngrids, nao, nao], order='C')
    int3cip2_pinned_buf = cp.cuda.alloc_pinned_memory(buf_size * 8)
    int3c_ip2 = np.frombuffer(int3cip2_pinned_buf, np.float64, buf_size).reshape([3, ngrids, nao, nao], order='C')

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        int3c_grid_slice = cp.zeros([6, p1-p0, nao, nao], order='C')
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

            int3c_angular_slice = cp.zeros([6, p1-p0, j1-j0, i1-i0], order='C')

            charge_exponents_pointer = c_null_ptr()
            if charge_exponents is not None:
                exponents_slice = charge_exponents[p0:p1]
                charge_exponents_pointer = exponents_slice.data.ptr

            grids_slice = grids[p0:p1, :]
            err = libgint.GINTfill_int3c1e_ip(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(grids_slice.data.ptr, ctypes.c_void_p),
                ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
                ctypes.c_int(p1-p0),
                ctypes.cast(int3c_angular_slice.data.ptr, ctypes.c_void_p),
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

            int3c_grid_slice[:, :, i0:i1, j0:j1] = int3c_angular_slice.transpose(0,1,3,2)

        ao_idx = np.argsort(intopt._ao_idx)
        grid_idx = np.arange(p1-p0)
        derivative_idx = np.arange(6)
        int3c_grid_slice = int3c_grid_slice[np.ix_(derivative_idx, grid_idx, ao_idx, ao_idx)]

        # Each piece of the following memory is contiguous
        int3c_grid_slice[0, :, :, :].get(out = int3c_ip1[0, p0:p1, :, :])
        int3c_grid_slice[1, :, :, :].get(out = int3c_ip1[1, p0:p1, :, :])
        int3c_grid_slice[2, :, :, :].get(out = int3c_ip1[2, p0:p1, :, :])
        int3c_grid_slice[3, :, :, :].get(out = int3c_ip2[0, p0:p1, :, :])
        int3c_grid_slice[4, :, :, :].get(out = int3c_ip2[1, p0:p1, :, :])
        int3c_grid_slice[5, :, :, :].get(out = int3c_ip2[2, p0:p1, :, :])

    return int3c_ip1, int3c_ip2

def get_int3c1e_ip1_charge_contracted(mol, grids, charge_exponents, charges, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]
    charges = cp.asarray(charges).astype(np.float64)

    charges = charges.reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)

    int1e_charge_contracted = cp.empty([3, mol.nao, mol.nao], order='C')
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
        n_charge_sum_per_thread = 10

        int1e_angular_slice = cp.zeros([3, j1-j0, i1-i0], order='C')

        err = libgint.GINTfill_int3c1e_ip1_charge_contracted(
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
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=1, ang=lj)
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=2, ang=li)

        int1e_charge_contracted[:, i0:i1, j0:j1] = int1e_angular_slice.transpose(0,2,1)

    return intopt.unsort_orbitals(int1e_charge_contracted, axis=[1,2])

def get_int3c1e_ip1_density_contracted(mol, grids, charge_exponents, dm, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

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

    nao = intopt._sorted_mol.nao

    i_atom_of_each_shell = intopt._sorted_mol._bas[:, ATOM_OF]
    i_atom_of_each_shell = cp.array(i_atom_of_each_shell, dtype=np.int32)

    ip1_per_atom = cp.zeros([mol.natm, 3, ngrids])

    for cp_ij_id, _ in enumerate(intopt.log_qs):
        stream = cp.cuda.get_current_stream()

        log_q_ij = intopt.log_qs[cp_ij_id]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        charge_exponents_pointer = c_null_ptr()
        if charge_exponents is not None:
            charge_exponents_pointer = charge_exponents.data.ptr

        err = libgint.GINTfill_int3c1e_ip1_density_contracted(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(grids.data.ptr, ctypes.c_void_p),
            ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.cast(ip1_per_atom.data.ptr, ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.cast(i_atom_of_each_shell.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nao),
            ctypes.c_double(omega))

        if err != 0:
            raise RuntimeError('GINTfill_int3c1e_charge_contracted failed')

    return ip1_per_atom

def get_int3c1e_ip2_density_contracted(mol, grids, charge_exponents, dm, intopt):
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

    int3c_density_contracted = cp.zeros([3, ngrids], order='C')

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

            err = libgint.GINTfill_int3c1e_ip2_density_contracted(
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

    return int3c_density_contracted

def get_int3c1e_ip2_charge_contracted(mol, grids, charge_exponents, charges, gridslice, output, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    ngrids = grids.shape[0]
    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]
    charges = cp.asarray(charges).astype(np.float64)

    charges = charges.reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)

    n_atom = len(gridslice)
    i_atom_of_each_charge = [[i_atom] * (gridslice[i_atom][1] - gridslice[i_atom][0]) for i_atom in range(n_atom)]
    i_atom_of_each_charge = sum(i_atom_of_each_charge, [])
    i_atom_of_each_charge = cp.array(i_atom_of_each_charge, dtype=np.int32)

    assert isinstance(output, cp.ndarray)
    assert output.shape == (n_atom, 3, mol.nao, mol.nao)

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

        int1e_angular_slice = cp.zeros([n_atom, 3, j1-j0, i1-i0], order='C')

        err = libgint.GINTfill_int3c1e_ip2_charge_contracted(
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
            ctypes.cast(i_atom_of_each_charge.data.ptr, ctypes.c_void_p),
            ctypes.c_double(omega))

        if err != 0:
            raise RuntimeError('GINTfill_int3c1e_charge_contracted failed')

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        if not mol.cart:
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=2, ang=lj)
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=3, ang=li)

        output[np.ix_(range(n_atom), range(3), intopt._ao_idx[i0:i1], intopt._ao_idx[j0:j1])] += int1e_angular_slice.transpose(0,1,3,2)

def get_int3c1e_ip1_charge_and_density_contracted(mol, grids, charge_exponents, dm, charges, intopt):
    dm = cp.asarray(dm)
    if dm.ndim == 3:
        if dm.shape[0] > 2:
            print("Warning: more than two density matrices are found for int3c1e kernel. "
                  "They will be summed up to one density matrix.")
        dm = cp.einsum("ijk->jk", dm)

    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1] and dm.shape[0] == mol.nao

    int3c_ip1 = get_int3c1e_ip1_charge_contracted(mol, grids, charge_exponents, charges, intopt)
    int3c_ip1 = cp.einsum('xij,ij->xi', int3c_ip1, dm)
    return int3c_ip1

def get_int3c1e_ip2_charge_and_density_contracted(mol, grids, charge_exponents, dm, charges, intopt):
    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]
    charges = cp.asarray(charges).astype(np.float64)

    int3c_ip2 = get_int3c1e_ip2_density_contracted(mol, grids, charge_exponents, dm, intopt)
    int3c_ip2 = int3c_ip2 * charges
    return int3c_ip2

def int1e_grids_ip1(mol, grids, charge_exponents=None, dm=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    r'''
    This function computes
    $$\left(\frac{\partial}{\partial \vec{A}} \mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    where $\mu(\vec{r})$ centers at $\vec{A}$ and $\nu(\vec{r})$ centers at $\vec{B}$.

    If charges is not None and density is None, the function computes the following contraction:
    $$\sum_{C}^{n_{charge}} q_C \left(\frac{\partial}{\partial \vec{A}} \mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    where $q_C$ is the charge centered at $\vec{C}$.

    If charges is not None and dm is not None, the function computes the following contraction:
    $$\sum_\nu^{n_{ao}} D_{\mu\nu} \sum_{C}^{n_{charge}} q_C
        \left(\frac{\partial}{\partial \vec{A}} \mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$

    If dm is not None and charges is None, the function computes the following contraction:
    $$\sum_{\mu \in \{\text{AO of atom A}\}} \sum_\nu^{n_{ao}} D_{\mu\nu}
        \left(\frac{\partial}{\partial \vec{A}} \mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    The output dimension is $(n_{atom}, 3, n_{charge})$.
    '''
    assert grids is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    if dm is None and charges is None:
        return get_int3c1e_ip(mol, grids, charge_exponents, intopt)[0]
    elif charges is not None:
        if dm is not None:
            return get_int3c1e_ip1_charge_and_density_contracted(mol, grids, charge_exponents, dm, charges, intopt)
        else:
            return get_int3c1e_ip1_charge_contracted(mol, grids, charge_exponents, charges, intopt)
    else:
        assert dm is not None
        return get_int3c1e_ip1_density_contracted(mol, grids, charge_exponents, dm, intopt)

def int1e_grids_ip2(mol, grids, charge_exponents=None, dm=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    r'''
    This function computes
    $$\left(\mu \middle| \frac{\partial}{\partial \vec{C}} \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    where $\mu(\vec{r})$ centers at $\vec{A}$ and $\nu(\vec{r})$ centers at $\vec{B}$.

    If dm is not None and charges is None, the function computes the following contraction:
    $$\sum_{\mu, \nu}^{n_{ao}} D_{\mu\nu} \left(\mu \middle| \frac{\partial}{\partial \vec{C}} \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$

    If dm is not None and charges is not None, the function computes the following contraction:
    $$q_C \sum_{\mu, \nu}^{n_{ao}} D_{\mu\nu} \left(\mu \middle| \frac{\partial}{\partial \vec{C}} \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    where $q_C$ is the charge centered at $\vec{C}$.

    If charges is not None and dm is None, the function computes the following contraction:
    $$\sum_{C}^{n_{charge}} q_C \left(\mu \middle| \frac{\partial}{\partial \vec{C}} \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    Notice that this summation should not be performed if the charges originates from different atomic centers.
    '''
    assert grids is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    if dm is None and charges is None:
        return get_int3c1e_ip(mol, grids, charge_exponents, intopt)[1]
    elif dm is not None:
        if charges is not None:
            return get_int3c1e_ip2_charge_and_density_contracted(mol, grids, charge_exponents, dm, charges, intopt)
        else:
            return get_int3c1e_ip2_density_contracted(mol, grids, charge_exponents, dm, intopt)
    else:
        assert charges is not None
        output = cp.zeros([1, 3, mol.nao, mol.nao])
        get_int3c1e_ip2_charge_contracted(mol, grids, charge_exponents, charges, [[0, grids.shape[0]]], output, intopt)
        return output.reshape([3, mol.nao, mol.nao])

def int1e_grids_ip2_charge_contracted(mol, grids, charges, gridslice, output, charge_exponents=None, direct_scf_tol=1e-13, intopt=None):
    r'''
    This function computes the following contraction:
    $$\sum_{C \in \{\text{grid attached to atom A}\}} q_C
        \left(\mu \middle| \frac{\partial}{\partial \vec{C}} \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    where $q_C$ is the charge centered at $\vec{C}$. The output dimension is $(n_{atom}, 3, n_{ao}, n_{ao})$.
    '''
    assert grids is not None
    assert charges is not None
    assert gridslice is not None
    assert output is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=False)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert not intopt.aosym

    return get_int3c1e_ip2_charge_contracted(mol, grids, charge_exponents, charges, gridslice, output, intopt)
