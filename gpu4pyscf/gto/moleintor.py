# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import load_library, cart2sph, block_c2s_diag, get_avail_mem
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.int4c2e import BasisProdCache
from gpu4pyscf.df.int3c2e import sort_mol, _split_l_ctr_groups, get_pairing
from gpu4pyscf.gto.mole import basis_seg_contraction

BLKSIZE = 128

libgvhf = load_library('libgvhf')
libgint = load_library('libgint')

class VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        # use local basis_seg_contraction for efficiency
        # TODO: switch _mol and mol
        self.mol = basis_seg_contraction(mol,allow_replica=True)
        self._mol = mol

        '''
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        '''
        self.nao = self.mol.nao

        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

        self.bpcache = None

        self.cart_ao_idx = None
        self.sph_ao_idx = None

        self.cart_ao_loc = []
        self.sph_ao_loc = []
        self.cart2sph = None

        self.angular = None

        self.cp_idx = None
        self.cp_jdx = None

        self.log_qs = None

    init_cvhf_direct = _vhf.VHFOpt.init_cvhf_direct

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        if self.bpcache is not None:
            libgvhf.GINTdel_basis_prod(ctypes.byref(self.bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

    def build(self, cutoff=1e-14, group_size=None, diag_block_with_triu=False, aosym=False):
        _mol = self._mol
        mol = self.mol

        log = logger.new_logger(_mol, _mol.verbose)
        cput0 = log.init_timer()
        sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)
        self.sorted_mol = sorted_mol
        if group_size is not None :
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size)
        self.nctr = len(uniq_l_ctr)

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, sorted_mol, self._intor, self._prescreen,
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
        cput1 = log.timer_debug1('Get pairing', *cput1)

        # contraction coefficient for ao basis
        cart_ao_loc = sorted_mol.ao_loc_nr(cart=True)
        sph_ao_loc = sorted_mol.ao_loc_nr(cart=False)
        self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
        self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
        self.angular = [l[0] for l in uniq_l_ctr]

        cart_ao_loc = mol.ao_loc_nr(cart=True)
        sph_ao_loc = mol.ao_loc_nr(cart=False)
        nao = sph_ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), sph_ao_loc[1:-1])
        self.sph_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

        # cartesian ao index
        nao = cart_ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), cart_ao_loc[1:-1])
        self.cart_ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        ncart = cart_ao_loc[-1]
        nsph = sph_ao_loc[-1]
        self.cart2sph = block_c2s_diag(self.angular, l_ctr_counts)
        cput1 = log.timer_debug1('AO cart2sph coeff', *cput1)

        if _mol.cart:
            inv_idx = np.argsort(self.cart_ao_idx, kind='stable').astype(np.int32)
            self.coeff = cp.eye(ncart)[:,inv_idx]
        else:
            inv_idx = np.argsort(self.sph_ao_idx, kind='stable').astype(np.int32)
            self.coeff = self.cart2sph[:, inv_idx]
        cput1 = log.timer_debug1('AO cart2sph coeff', *cput1)

        ao_loc = sorted_mol.ao_loc_nr(cart=True)
        cput1 = log.timer_debug1('Get AO pairs', *cput1)

        self.pair2bra = pair2bra
        self.pair2ket = pair2ket
        self.l_ctr_offsets = l_ctr_offsets
        bas_pair2shls = np.hstack(pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
        bas_pairs_locs = np.append(0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        self.bas_pair2shls = bas_pair2shls
        self.bas_pairs_locs = bas_pairs_locs
        ncptype = len(log_qs)

        cp_idx, cp_jdx = np.tril_indices(len(uniq_l_ctr))
        l_ij = list(zip(uniq_l_ctr[cp_idx, 0], uniq_l_ctr[cp_jdx, 0]))
        self.l_ij = np.asarray(l_ij)
        def get_n_hermite_density_of_angular_pair(l): return (l + 1) * (l + 2) * (l + 3) // 6
        n_density_per_pair = np.array([ get_n_hermite_density_of_angular_pair(li + lj) for (li, lj) in l_ij ])
        n_density_per_angular_pair = (bas_pairs_locs[1:] - bas_pairs_locs[:-1]) * n_density_per_pair
        self.density_offset = np.append(0, np.cumsum(n_density_per_angular_pair)).astype(np.int32)

        self.bpcache = ctypes.POINTER(BasisProdCache)()
        scale_shellpair_diag = 1.0
        libgint.GINTinit_basis_prod(
            ctypes.byref(self.bpcache), ctypes.c_double(scale_shellpair_diag),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
            sorted_mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(sorted_mol.natm),
            sorted_mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(sorted_mol.nbas),
            sorted_mol._env.ctypes.data_as(ctypes.c_void_p))

        cput1 = log.timer_debug1('Initialize GPU cache', *cput1)
        ncptype = len(self.log_qs)
        self.aosym = aosym
        if aosym:
            self.cp_idx, self.cp_jdx = np.tril_indices(ncptype)
        else:
            nl = int(round(np.sqrt(ncptype)))
            self.cp_idx, self.cp_jdx = np.unravel_index(np.arange(ncptype), (nl, nl))

        if _mol.cart:
            self.ao_loc = self.cart_ao_loc
            self.ao_idx = self.cart_ao_idx
        else:
            self.ao_loc = self.sph_ao_loc
            self.ao_idx = self.sph_ao_idx
# end of class VHFOpt


def get_int3c1e_slice(intopt, cp_ij_id, grids, out, omega):
    stream = cp.cuda.get_current_stream()
    if omega is None: omega = 0.0
    nao_cart = intopt.mol.nao

    cpi = intopt.cp_idx[cp_ij_id]
    cpj = intopt.cp_jdx[cp_ij_id]

    log_q_ij = intopt.log_qs[cp_ij_id]

    nbins = 1
    bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

    i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
    j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
    ni = i1 - i0
    nj = j1 - j0

    ao_offsets = np.array([i0, j0], dtype=np.int32)
    strides = np.array([ni, ni*nj], dtype=np.int32)

    err = libgint.GINTfill_int3c1e(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        intopt.bpcache,
        ctypes.cast(grids.data.ptr, ctypes.c_void_p),
        ctypes.c_int(grids.shape[0]),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao_cart),
        strides.ctypes.data_as(ctypes.c_void_p),
        ao_offsets.ctypes.data_as(ctypes.c_void_p),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbins),
        ctypes.c_int(cp_ij_id),
        ctypes.c_double(omega))

    if err != 0:
        raise RuntimeError('GINTfill_int3c1e failed')

def get_int3c1e(mol, grids, direct_scf_tol, omega):
    intopt = VHFOpt(mol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True, aosym=True, group_size=BLKSIZE)

    nao = mol.nao
    ngrids = grids.shape[0]
    total_double_number = ngrids * nao * nao
    cp.get_default_memory_pool().free_all_blocks()
    avail_mem = get_avail_mem()
    reserved_available_memory = avail_mem // 4 # Leave space for further allocations
    allowed_double_number = reserved_available_memory // 8
    n_grid_split = int(np.ceil(total_double_number / allowed_double_number))
    if (n_grid_split > 100):
        raise Exception(f"Available GPU memory ({avail_mem / 1e9 : .1f} GB) is too small for the 3 center integral, which requires {total_double_number * 8 / 1e9 : .1f} GB of memory")
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    int3c_pinned_memory_pool = cp.cuda.alloc_pinned_memory(ngrids * nao * nao * np.array([1.0]).nbytes)
    int3c = np.frombuffer(int3c_pinned_memory_pool, np.float64, ngrids * nao * nao).reshape([ngrids, nao, nao], order='C')
    # int3c = np.zeros([ngrids, nao, nao], order='C') # Using unpinned (pageable) memory, each memcpy is much slower, but there's no initialization time

    grids = cp.asarray(grids, order='C')

    for i_grid_split in range(0, ngrids, ngrids_per_split):
        ngrids_of_split = np.min([ngrids_per_split, ngrids - i_grid_split])
        int3c_grid_slice = cp.zeros([ngrids_of_split, nao, nao], order='C')
        for cp_ij_id, _ in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]
            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]

            int3c_angular_slice = cp.zeros([ngrids_of_split, j1-j0, i1-i0], order='C')
            get_int3c1e_slice(intopt, cp_ij_id, grids[i_grid_split : i_grid_split + ngrids_of_split], out=int3c_angular_slice, omega=omega)
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
            if not mol.cart:
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=1, ang=lj)
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=2, ang=li)
            int3c_grid_slice[:, j0:j1, i0:i1] = int3c_angular_slice
        row, col = np.tril_indices(nao)
        int3c_grid_slice[:, row, col] = int3c_grid_slice[:, col, row]
        ao_idx = np.argsort(intopt.ao_idx)
        grid_idx = np.arange(ngrids_of_split)
        int3c_grid_slice = int3c_grid_slice[np.ix_(grid_idx, ao_idx, ao_idx)]

        cp.cuda.runtime.memcpy(int3c[i_grid_split : i_grid_split + ngrids_of_split, :, :].ctypes.data, int3c_grid_slice.data.ptr, int3c_grid_slice.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
        # int3c[i_grid_split : i_grid_split + ngrids_of_split, :, :] = cp.asnumpy(int3c_grid_slice) # This is certainly the wrong way of DtoH memcpy

    return int3c

def get_int3c1e_density_contracted(mol, grids, dm, direct_scf_tol, omega):
    if cp.get_array_module(dm) is cp:
        dm = cp.asnumpy(dm)
    assert cp.get_array_module(dm) is np
    assert dm.ndim == 2

    intopt = VHFOpt(mol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=False, aosym=True, group_size=BLKSIZE)

    nao_cart = intopt.mol.nao
    ngrids = grids.shape[0]
    # TODO: Split ngrids to make sure GPU block and thread doesn't overflow

    dm = dm[np.ix_(intopt.ao_idx, intopt.ao_idx)] # intopt.ao_idx is in spherical basis
    if not mol.cart:
        cart2sph_transformation_matrix = cp.asnumpy(intopt.cart2sph)
        # TODO: This part is inefficient (O(N^3)), should be changed to the O(N^2) algorithm
        dm = cart2sph_transformation_matrix @ dm @ cart2sph_transformation_matrix.T
    ao_loc_sorted_order = intopt.sorted_mol.ao_loc_nr(cart = True) # This ao_loc order is consistent with the density matrix order and intopt.bas_pair2shls order

    l_ij = intopt.l_ij.T.flatten()
    n_total_hermite_density = intopt.density_offset[-1]
    dm_pair_ordered = np.zeros(n_total_hermite_density)
    dm = dm.flatten(order='F') # Column major order matches (i + j * n_ao) access pattern in the following function
    libgvhf.GINTinit_J_density_rys_preprocess(dm.ctypes.data_as(ctypes.c_void_p),
                                              dm_pair_ordered.ctypes.data_as(ctypes.c_void_p),
                                              ctypes.c_int(1), ctypes.c_int(nao_cart), ctypes.c_int(len(intopt.bas_pairs_locs) - 1),
                                              intopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                                              intopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                                              l_ij.ctypes.data_as(ctypes.c_void_p),
                                              intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
                                              ao_loc_sorted_order.ctypes.data_as(ctypes.c_void_p),
                                              intopt.bpcache)

    dm_pair_ordered = cp.asarray(dm_pair_ordered)
    grids = cp.asarray(grids, order='C')
    int3c_density_contracted = cp.zeros(ngrids)

    for cp_ij_id, _ in enumerate(intopt.log_qs):
        stream = cp.cuda.get_current_stream()
        if omega is None: omega = 0.0

        log_q_ij = intopt.log_qs[cp_ij_id]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        err = libgint.GINTfill_int3c1e_density_contracted(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(grids.data.ptr, ctypes.c_void_p),
            ctypes.c_int(grids.shape[0]),
            ctypes.cast(dm_pair_ordered.data.ptr, ctypes.c_void_p),
            intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(int3c_density_contracted.data.ptr, ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.c_double(omega))

        if err != 0:
            raise RuntimeError('GINTfill_int3c1e failed')

    return cp.asnumpy(int3c_density_contracted)

def intor(mol, intor, grids, dm=None, charges=None, direct_scf_tol=1e-13, omega=None):
    assert intor == 'int1e_grids' and grids is not None
    assert dm is None or charges is None, "Are you sure you want to contract the one electron integrals with both charge and density? If so, pass in density, obtain the result with n_charge and contract with the charges yourself."

    if dm is None and charges is None:
        return get_int3c1e(mol, grids, direct_scf_tol, omega)
    elif dm is not None:
        return get_int3c1e_density_contracted(mol, grids, dm, direct_scf_tol, omega)
    elif charges is not None:
        raise NotImplementedError()
    else:
        raise ValueError(f"Logic error in {__file__} {__name__}")
