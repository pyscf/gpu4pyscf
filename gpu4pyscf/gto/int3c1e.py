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
from pyscf.scf import _vhf
from pyscf.gto import ATOM_OF
from pyscf.lib import c_null_ptr
from gpu4pyscf.lib.cupy_helper import load_library, cart2sph, block_c2s_diag, get_avail_mem
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.int4c2e import BasisProdCache
from gpu4pyscf.df.int3c2e import sort_mol, _split_l_ctr_groups, get_pairing
from gpu4pyscf.gto.mole import basis_seg_contraction
from gpu4pyscf.__config__ import num_devices, _streams

BLKSIZE = 128

libgint = load_library('libgint')

class VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, intor='int2e', prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol = mol
        self._sorted_mol = None

        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

        self.cart_ao_loc = []
        self.sph_ao_loc = []

        self.angular = None

        self.cp_idx = None
        self.cp_jdx = None

        self.log_qs = None

    init_cvhf_direct = _vhf.VHFOpt.init_cvhf_direct

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        for n, bpcache in self._bpcache.items():
            libgint.GINTdel_basis_prod(ctypes.byref(bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

    def build(self, cutoff=1e-13, group_size=BLKSIZE, diag_block_with_triu=False, aosym=True):
        original_mol = self.mol
        mol = basis_seg_contraction(original_mol, allow_replica=True)[0]

        log = logger.new_logger(original_mol, original_mol.verbose)
        cput0 = log.init_timer()
        _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)
        self._sorted_mol = _sorted_mol

        if group_size is not None :
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size)
        self.l_ctr_counts = l_ctr_counts

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, _sorted_mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        q_cond = self.get_q_cond()
        cput1 = log.timer_debug1('Initialize q_cond', *cput0)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        log_qs, pair2bra, pair2ket = get_pairing(
            l_ctr_offsets, l_ctr_offsets, q_cond,
            diag_block_with_triu=diag_block_with_triu, aosym=aosym)
        self.log_qs = log_qs.copy()
        cput1 = log.timer_debug1('Get AO pairing', *cput1)

        # contraction coefficient for ao basis
        cart_ao_loc = _sorted_mol.ao_loc_nr(cart=True)
        sph_ao_loc = _sorted_mol.ao_loc_nr(cart=False)
        self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
        self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
        self.angular = [l[0] for l in uniq_l_ctr]

        # Sorted AO indices
        ao_loc = mol.ao_loc_nr(cart=original_mol.cart)
        ao_idx = np.array_split(np.arange(original_mol.nao), ao_loc[1:-1])
        self._ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        cput1 = log.timer_debug1('AO indices', *cput1)

        ao_loc = cart_ao_loc

        self.pair2bra = pair2bra
        self.pair2ket = pair2ket
        self.l_ctr_offsets = l_ctr_offsets
        bas_pair2shls = np.hstack(pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
        bas_pairs_locs = np.append(0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        self.bas_pair2shls = bas_pair2shls
        self.bas_pairs_locs = bas_pairs_locs
        ncptype = len(self.log_qs)

        n_uniq_l_ctr = len(uniq_l_ctr)
        if aosym:
            # This symmetric case is different from self.cp_idx, whose length is ncptype = n_uniq_l_ctr ** 2
            cp_idx, cp_jdx = np.tril_indices(n_uniq_l_ctr)
        else:
            cp_idx, cp_jdx = np.unravel_index(np.arange(n_uniq_l_ctr**2), (n_uniq_l_ctr, n_uniq_l_ctr))
        l_ij = list(zip(uniq_l_ctr[cp_idx, 0], uniq_l_ctr[cp_jdx, 0]))
        self.l_ij = np.asarray(l_ij)
        def get_n_hermite_density_of_angular_pair(l):
            return (l + 1) * (l + 2) * (l + 3) // 6
        n_density_per_pair = np.array([ get_n_hermite_density_of_angular_pair(li + lj) for (li, lj) in l_ij ])
        n_density_per_angular_pair = (bas_pairs_locs[1:] - bas_pairs_locs[:-1]) * n_density_per_pair
        self.density_offset = np.append(0, np.cumsum(n_density_per_angular_pair)).astype(np.int32)

        self._bpcache = {}
        for n in range(num_devices):
            with cp.cuda.Device(n), _streams[n]:
                bpcache = ctypes.POINTER(BasisProdCache)()
                scale_shellpair_diag = 1.0
                libgint.GINTinit_basis_prod(
                    ctypes.byref(bpcache), ctypes.c_double(scale_shellpair_diag),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                    bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
                    _sorted_mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol.natm),
                    _sorted_mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol.nbas),
                    _sorted_mol._env.ctypes.data_as(ctypes.c_void_p))
                self._bpcache[n] = bpcache

        cput1 = log.timer_debug1('Initialize GPU cache', *cput1)
        self.aosym = aosym
        if aosym:
            self.cp_idx, self.cp_jdx = np.tril_indices(ncptype)
        else:
            nl = int(round(np.sqrt(ncptype)))
            self.cp_idx, self.cp_jdx = np.unravel_index(np.arange(ncptype), (nl, nl))

        if original_mol.cart:
            self.ao_loc = self.cart_ao_loc
        else:
            self.ao_loc = self.sph_ao_loc

    def sort_orbitals(self, mat, axis=[]):
        ''' Transform given axis of a matrix into sorted AO,
        '''
        idx = self._ao_idx
        shape_ones = (1,) * mat.ndim
        fancy_index = []
        for dim, n in enumerate(mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[]):
        ''' Transform given axis of a matrix into sorted AO,
        '''
        idx = self._ao_idx
        shape_ones = (1,) * sorted_mat.ndim
        fancy_index = []
        for dim, n in enumerate(sorted_mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        mat = cp.empty_like(sorted_mat)
        mat[tuple(fancy_index)] = sorted_mat
        return mat

    @property
    def bpcache(self):
        device_id = cp.cuda.Device().id
        bpcache = self._bpcache[device_id]
        return bpcache

    @property
    def cart2sph(self):
        return block_c2s_diag(self.angular, self.l_ctr_counts)
# end of class VHFOpt


def get_int3c1e(mol, grids, charge_exponents, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    nao = mol.nao
    ngrids = grids.shape[0]
    total_double_number = ngrids * nao * nao
    cp.get_default_memory_pool().free_all_blocks()
    avail_mem = get_avail_mem()
    reserved_available_memory = avail_mem // 4 # Leave space for further allocations
    allowed_double_number = reserved_available_memory // 8
    n_grid_split = int(np.ceil(total_double_number / allowed_double_number))
    if (n_grid_split > 100):
        raise Exception(f"Available GPU memory ({avail_mem / 1e9 : .1f} GB) is too small for the 3 center integral, "
                        "which requires {total_double_number * 8 / 1e9 : .1f} GB of memory")
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    buf_size = ngrids * nao * nao
    int3c_pinned_buf = cp.cuda.alloc_pinned_memory(buf_size * 8)
    int3c = np.frombuffer(int3c_pinned_buf, np.float64, buf_size).reshape([ngrids, nao, nao], order='C')
    # int3c = np.zeros([ngrids, nao, nao], order='C') # Using unpinned (pageable) memory, each memcpy is much slower, but there's no initialization time

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        int3c_grid_slice = cp.zeros([p1-p0, nao, nao], order='C')
        for cp_ij_id, _ in enumerate(intopt.log_qs):
            log_q_ij = intopt.log_qs[cp_ij_id]
            if len(log_q_ij) == 0:
                continue

            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            stream = cp.cuda.get_current_stream()

            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            ni = i1 - i0
            nj = j1 - j0

            ao_offsets = np.array([i0, j0], dtype=np.int32)
            strides = np.array([ni, ni*nj], dtype=np.int32)

            int3c_angular_slice = cp.zeros([p1-p0, j1-j0, i1-i0], order='C')

            charge_exponents_pointer = c_null_ptr()
            if charge_exponents is not None:
                exponents_slice = charge_exponents[p0:p1]
                charge_exponents_pointer = exponents_slice.data.ptr
            grids_slice = grids[p0:p1]
            err = libgint.GINTfill_int3c1e(
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
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=1, ang=lj)
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=2, ang=li)

            int3c_grid_slice[:, j0:j1, i0:i1] = int3c_angular_slice

        row, col = np.tril_indices(nao)
        int3c_grid_slice[:, row, col] = int3c_grid_slice[:, col, row]
        #ao_idx = np.argsort(intopt._ao_idx)
        #grid_idx = np.arange(p1-p0)
        #int3c_grid_slice = int3c_grid_slice[np.ix_(grid_idx, ao_idx, ao_idx)]
        int3c_grid_slice = intopt.unsort_orbitals(int3c_grid_slice, axis=[1,2])
        int3c_grid_slice.get(out = int3c[p0:p1, :, :])

    return int3c

def get_int3c1e_charge_contracted(mol, grids, charge_exponents, charges, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    nao = mol.nao

    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]

    grids = cp.asarray(grids, order='C')
    charges = cp.asarray(charges).astype(np.float64)

    charges = charges.reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    int1e_charge_contracted = cp.zeros([mol.nao, mol.nao], order='C')
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

        int1e_angular_slice = cp.zeros([j1-j0, i1-i0], order='C')

        err = libgint.GINTfill_int3c1e_charge_contracted(
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
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=0, ang=lj)
            int1e_angular_slice = cart2sph(int1e_angular_slice, axis=1, ang=li)

        int1e_charge_contracted[j0:j1, i0:i1] = int1e_angular_slice

    row, col = np.tril_indices(nao)
    int1e_charge_contracted[row, col] = int1e_charge_contracted[col, row]
    #ao_idx = np.argsort(intopt._ao_idx)
    #int1e_charge_contracted = int1e_charge_contracted[np.ix_(ao_idx, ao_idx)]
    int1e_charge_contracted = intopt.unsort_orbitals(int1e_charge_contracted, axis=[0,1])
    return int1e_charge_contracted

def get_int3c1e_density_contracted(mol, grids, charge_exponents, dm, intopt):
    omega = mol.omega
    assert omega >= 0.0, "Short-range one electron integrals with GPU acceleration is not implemented."

    dm = cp.asarray(dm)
    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1] and dm.shape[0] == mol.nao

    nao_cart = intopt._sorted_mol.nao
    ngrids = grids.shape[0]

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
                                              ctypes.c_int(1), ctypes.c_int(nao_cart), ctypes.c_int(len(intopt.bas_pairs_locs) - 1),
                                              intopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                                              intopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                                              l_ij.ctypes.data_as(ctypes.c_void_p),
                                              intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
                                              ao_loc_sorted_order.ctypes.data_as(ctypes.c_void_p),
                                              bas_coords.ctypes.data_as(ctypes.c_void_p),
                                              ctypes.c_bool(True))

    dm_pair_ordered = cp.asarray(dm_pair_ordered)

    n_threads_per_block_1d = 16
    n_max_blocks_per_grid_1d = 65535
    n_max_threads_1d = n_threads_per_block_1d * n_max_blocks_per_grid_1d
    n_grid_split = int(np.ceil(ngrids / n_max_threads_1d))
    if (n_grid_split > 100):
        print(f"Grid dimension = {ngrids} is too large, more than 100 kernels for one electron integral will be launched.")
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')

    int3c_density_contracted = cp.zeros(ngrids)

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        for cp_ij_id, _ in enumerate(intopt.log_qs):
            log_q_ij = intopt.log_qs[cp_ij_id]
            if len(log_q_ij) == 0:
                continue

            stream = cp.cuda.get_current_stream()

            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

            charge_exponents_pointer = c_null_ptr()
            if charge_exponents is not None:
                exponents_slice = charge_exponents[p0:p1]
                charge_exponents_pointer = exponents_slice.data.ptr

            # n_pair_sum_per_thread = 1 # means every thread processes one pair and one grid
            # n_pair_sum_per_thread = nao_cart # or larger number gaurantees one thread processes one grid and all pairs of the same type
            n_pair_sum_per_thread = nao_cart
            grids_slice = grids[p0:p1, :]
            err = libgint.GINTfill_int3c1e_density_contracted(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(grids_slice.data.ptr, ctypes.c_void_p),
                ctypes.cast(charge_exponents_pointer, ctypes.c_void_p),
                ctypes.c_int(p1-p0),
                ctypes.cast(dm_pair_ordered.data.ptr, ctypes.c_void_p),
                intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
                ctypes.cast(int3c_density_contracted[p0:p1].data.ptr, ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(omega),
                ctypes.c_int(n_pair_sum_per_thread))

            if err != 0:
                raise RuntimeError('GINTfill_int3c1e_density_contracted failed')

    return int3c_density_contracted

def int1e_grids(mol, grids, charge_exponents=None, dm=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    r'''
    This function computes
    $$\left(\mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$

    If charges is not None, the function computes the following contraction:
    $$\sum_{C}^{n_{charge}} q_C \left(\mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    where $q_C$ is the charge centered at $\vec{C}$.

    If dm is not None, the function computes the following contraction:
    $$\sum_{\mu, \nu}^{n_{ao}} D_{\mu\nu} \left(\frac{\partial}{\partial \vec{A}} \mu \middle| \frac{1}{|\vec{r} - \vec{C}|} \middle| \nu\right)$$
    '''
    assert grids is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=True)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."
        assert intopt.aosym

    assert dm is None or charges is None, \
        "Are you sure you want to contract the one electron integrals with both charge and density? " + \
        "If so, pass in density, obtain the result with n_charge and contract with the charges yourself."

    if dm is None and charges is None:
        return get_int3c1e(mol, grids, charge_exponents, intopt)
    elif dm is not None:
        if dm.ndim == 2:
            return get_int3c1e_density_contracted(mol, grids, charge_exponents, dm, intopt)
        else:
            assert dm.ndim == 3
            n_dm = dm.shape[0]
            ngrids = grids.shape[0]
            if n_dm == 1:
                return get_int3c1e_density_contracted(mol, grids, charge_exponents, dm[0], intopt).reshape(1, ngrids)
            int3c_density_contracted = cp.empty((n_dm, ngrids))
            for i_dm in range(n_dm):
                int3c_density_contracted[i_dm] = get_int3c1e_density_contracted(mol, grids, charge_exponents, dm[i_dm], intopt)
            return int3c_density_contracted
    elif charges is not None:
        if charges.ndim == 1:
            return get_int3c1e_charge_contracted(mol, grids, charge_exponents, charges, intopt)
        else:
            assert charges.ndim == 2
            n_charges = charges.shape[0]
            nao = mol.nao
            if n_charges == 1:
                return get_int3c1e_charge_contracted(mol, grids, charge_exponents, charges[0], intopt).reshape(1, nao, nao)
            int3c_charge_contracted = cp.empty((n_charges, nao, nao))
            for i_charge in range(n_charges):
                int3c_charge_contracted[i_charge] = get_int3c1e_charge_contracted(mol, grids, charge_exponents, charges[i_charge], intopt)
            return int3c_charge_contracted
    else:
        raise ValueError(f"Logic error in {__file__} {__name__}")
