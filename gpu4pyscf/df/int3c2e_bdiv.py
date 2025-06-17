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

'''
3-center 2-electron Coulomb integral helper functions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, conc_env
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, dist_matrix, asarray)
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import g_pair_idx, _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.gto.mole import basis_seg_contraction, extract_pgto_params, cart2sph_by_l

__all__ = [
    'aux_e2',
]
libgint_rys = load_library('libgint_rys')
libgint_rys.fill_int3c2e.restype = ctypes.c_int
libgint_rys.fill_int3c2e_bdiv.restype = ctypes.c_int
libgint_rys.init_constant.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 45
THREADS = 256

def aux_e2(mol, auxmol):
    r'''
    3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    ao_pair_mapping = cp.asarray(int3c2e_opt.create_ao_pair_mapping())
    nao, nao_orig = int3c2e_opt.coeff.shape
    naux = int3c2e_opt.aux_coeff.shape[0]
    out = cp.zeros((nao*nao, naux))
    p0 = p1 = 0
    for ij_shls, eri3c in int3c2e_opt.int3c2e_generator():
        p0, p1 = p1, p1 + eri3c.shape[0]
        addr = ao_pair_mapping[p0:p1]
        out[addr] = eri3c
        i, j = divmod(addr, nao)
        out[j*nao+i] = eri3c
    log = logger.new_logger(mol)
    t1 = log.init_timer()
    out = out.reshape(nao, nao, naux)
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    coeff = cp.asarray(int3c2e_opt.coeff)
    out = contract('pqr,rk->pqk', out, aux_coeff)
    out = contract('pqk,qj->pjk', out, coeff)
    out = contract('pjk,pi->ijk', out, coeff)
    t1 = log.timer_debug1('aux_e2: transform basis ordering', *t1)
    return out

def compressed_aux_e2(mol, auxmol):
    r'''
    Returns compressed_int3c, rows, cols. The compressed_int3c stores the
    3-center integrals (ij|k) compressed on the orbital-pair dimensions.
    The addresses of the non-zero pairs are stored in the rows and cols indices.
    The 3-center integral tensor can be restored by:
        int3c[rows,cols] = compressed_int3c 
    '''
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    eri3c = next(int3c2e_opt.int3c2e_bdiv_generator())
    eri3c = int3c2e_opt.orbital_pair_cart2sph(eri3c, inplace=True)
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    eri3c = eri3c.dot(aux_coeff)
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
    rows, cols = divmod(cp.asarray(ao_pair_mapping), mol.nao)
    # compressed = int3c[ao_idx[:,None],ao_idx][rows,cols]
    #            = int3c[ao_idx[rows],ao_idx[cols]]
    ao_idx = cp.asarray(int3c2e_opt.ao_idx)
    return eri3c, ao_idx[rows], ao_idx[cols]

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        self.auxmol = auxmol
        self.sorted_mol = None

    def build(self, cutoff=1e-14):
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        # allow_replica=True to transform the general contracted basis sets into
        # segment contracted sets
        mol, c2s = basis_seg_contraction(self.mol, allow_replica=True)
        mol, coeff, uniq_l_ctr, l_ctr_counts, bas_mapping = group_basis(
            mol, tile=1, return_bas_mapping=True)
        self.sorted_mol = mol
        self.uniq_l_ctr = uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.coeff = coeff.dot(c2s).get()
        # Sorted AO indices, allow using the fancyindices to transform tensors
        # between sorted_mol and mol (see function sort_orbitals)
        ao_loc = mol.ao_loc_nr(cart=self.mol.cart)
        ao_idx = np.array_split(np.arange(self.mol.nao), ao_loc[1:-1])
        self.ao_idx = np.hstack([ao_idx[i] for i in bas_mapping]).argsort()

        auxmol, coeff, uniq_l_ctr_aux, l_ctr_aux_counts = group_basis(self.auxmol, tile=1)
        self.sorted_auxmol = auxmol
        self.uniq_l_ctr_aux = uniq_l_ctr_aux
        l_ctr_aux_offsets = self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_aux_counts))
        self.aux_coeff = coeff.get()

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[mol.nbas:,PTR_BAS_COORD] += off
        self._atm = _atm_cpu
        self._bas = _bas_cpu
        self._env = _env_cpu

        ao_loc_cpu = mol.ao_loc
        aux_loc = auxmol.ao_loc

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = cp.asarray(_conc_locs(ao_loc_cpu, aux_loc), dtype=np.int32)
        self.int3c2e_envs = Int3c2eEnvVars(
            mol.natm, mol.nbas, _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc.data.ptr, math.log(cutoff),
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        self.int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc)

        nksh_per_block = 16
        # the auxiliary function offset (address) in the output tensor for each blockIdx.y
        ksh_offsets = []
        for ksh0, ksh1 in zip(l_ctr_aux_offsets[:-1], l_ctr_aux_offsets[1:]):
            ksh_offsets.append(np.arange(ksh0, ksh1, nksh_per_block, dtype=np.int32))
        ksh_offsets.append(l_ctr_aux_offsets[-1])
        ksh_offsets = np.hstack(ksh_offsets)
        ksh_offsets += mol.nbas
        self.ksh_offsets = ksh_offsets

        uniq_l = uniq_l_ctr[:,0]
        assert uniq_l.max() <= LMAX
        n_groups = len(uniq_l)
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))

        ovlp = estimate_shl_ovlp(mol)
        mask = np.tril(ovlp > cutoff)
        # The effective shell pair = ish*nbas+jsh
        shl_pair_idx = []
        # the bas_ij_idx offset for each blockIdx.x
        shl_pair_offsets = []
        # the AO-pair offset (address) in the output tensor for each blockIdx.x
        ao_pair_loc = []
        nao_pair0 = nao_pair = 0
        sp0 = sp1 = 0
        nbas = mol.nbas
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish, jsh = cp.where(mask[ish0:ish1,jsh0:jsh1])
            if len(ish) == 0:
                continue
            ish += ish0
            jsh += jsh0
            idx = ish * nbas + jsh
            nshl_pair = idx.size
            shl_pair_idx.append(idx)
            nfi = (li + 1) * (li + 2) // 2
            nfj = (lj + 1) * (lj + 2) // 2
            nfij = nfi * nfj
            nao_pair0, nao_pair = nao_pair, nao_pair + nfij * nshl_pair

            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = _estimate_shl_pairs_per_block(li, lj, nshl_pair)
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))
            ao_pair_loc.append(
                np.arange(nao_pair0, nao_pair, nsp_per_block*nfij, dtype=np.int32))
            if log.verbose >= logger.DEBUG2:
                log.debug2('group=(%d,%d), li,lj=(%d,%d), sp range(%d,%d,%d), '
                           'nao_pair offset=%d',
                           i, j, li, lj, sp0, sp1, nsp_per_block, nao_pair0)

        self.shl_pair_idx = shl_pair_idx
        shl_pair_offsets.append([sp1])
        self.shl_pair_offsets = np.hstack(shl_pair_offsets)
        ao_pair_loc.append(nao_pair)
        self.ao_pair_loc = np.hstack(ao_pair_loc)
        if log.verbose >= logger.DEBUG1:
            log.timer_debug1('initialize int3c2e_kernel', *t0)
        return self

    def int3c2e_generator(self, cutoff=1e-14, verbose=None):
        '''Generator that yields the 3c2e integral tensor in multiple batches.

        Each batch is a two-dimensional tensor. The first dimension corresponds
        to compressed orbital pairs, which can be indexed using the row and cols
        returned by the .orbital_pair_nonzero_indices() method. The second
        dimension is a slice along the auxiliary basis dimension.
        '''
        if self.sorted_mol is None:
            self.build(cutoff)
        log = logger.new_logger(self.mol, verbose)
        t0 = t1 = log.init_timer()
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        int3c2e_envs = self.int3c2e_envs
        _atm_cpu = self._atm
        _bas_cpu = self._bas
        _env_cpu = self._env
        mol = self.sorted_mol
        aux_loc = self.sorted_auxmol.ao_loc
        naux = aux_loc[-1]

        uniq_l = self.uniq_l_ctr[:,0]
        nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
        n_groups = len(uniq_l)
        ij_tasks = [(i, j) for i in range(n_groups) for j in range(i+1)]
        npair_ij = 0
        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            nfij = nfcart[i] * nfcart[j]
            npair_ij = max(npair_ij, len(bas_ij_idx) * nfij)
        buf = cp.empty((npair_ij, naux))

        init_constant(mol)
        kern = libgint_rys.fill_int3c2e
        timing_collection = {}
        kern_counts = 0

        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            npair_ij = len(bas_ij_idx)
            if npair_ij == 0:
                continue
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            bas_ij_idx = cp.asarray(bas_ij_idx, dtype=np.int32)
            li = uniq_l[i]
            lj = uniq_l[j]
            nfij = nfcart[i] * nfcart[j]
            eri3c = cp.ndarray((npair_ij*nfij, naux), dtype=np.float64, memptr=buf.data)

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk)
                log.debug2('int3c2e_scheme for %s: %s', lll, scheme)
                err = kern(
                    ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice), aux_loc.ctypes,
                    ctypes.c_int(naux), ctypes.c_int(npair_ij),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(mol.natm),
                    _bas_cpu.ctypes, ctypes.c_int(mol.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {lll}', *t1), t1
                    if lll not in timing_collection:
                        timing_collection[lll] = 0
                    timing_collection[lll] += t1[1] - t1p[1]
                    kern_counts += 1

            ij_shls = ish0, ish1, jsh0, jsh1
            yield ij_shls, eri3c

        if log.verbose >= logger.DEBUG1:
            cp.cuda.Stream.null.synchronize()
            log.timer('int3c2e', *t0)
            log.debug1('kernel launches %d', kern_counts)
            for lll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', lll, t)

    def int3c2e_bdiv_generator(self, cutoff=1e-14, batch_size=None, verbose=None):
        '''An iterator to generate eri3c blocks using the block-divergent
        integral parallelism
        '''
        if self.sorted_mol is None:
            self.build(cutoff)
        log = logger.new_logger(self.mol, verbose)
        t0 = log.init_timer()
        int3c2e_envs = self.int3c2e_envs
        _atm_cpu = self._atm
        _bas_cpu = self._bas
        _env_cpu = self._env
        sorted_mol = self.sorted_mol
        aux_loc = self.sorted_auxmol.ao_loc
        nao_pair = self.ao_pair_loc[-1]

        # nst_lookup stores the nst_per_block for each (li,lj,lk) pattern
        nst_lookup = asarray(create_nst_lookup_table(), dtype=np.int32)

        shl_pair_idx = asarray(np.hstack(self.shl_pair_idx), dtype=np.int32)
        shl_pair_offsets = asarray(self.shl_pair_offsets, dtype=np.int32)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        ksh_offsets = self.ksh_offsets
        ksh_offsets_gpu = asarray(ksh_offsets, dtype=np.int32)
        ksh_blocks = len(ksh_offsets) - 1
        ao_pair_loc = asarray(self.ao_pair_loc, dtype=np.int32)
        log.debug1('sp_blocks = %d, ksh_blocks = %d', nbatches_shl_pair, ksh_blocks)

        # Group ksh_blocks into batches. Use ksh_block_partitions to index the
        # first ksh_block for each batch.
        aux_loc_by_block = aux_loc[ksh_offsets - sorted_mol.nbas]
        if batch_size is None:
            ksh_block_partitions = [0, ksh_blocks]
        else:
            ksh_block_partitions = group_blocks(aux_loc_by_block, batch_size)

        init_constant(sorted_mol)
        kern = libgint_rys.fill_int3c2e_bdiv
        for start, stop in zip(ksh_block_partitions[:-1], ksh_block_partitions[1:]):
            nblocks = stop - start
            ksh_offsets_batch = ksh_offsets_gpu[start:]
            k0 = aux_loc_by_block[start]
            k1 = aux_loc_by_block[stop]
            naux_batch = k1 - k0
            eri3c = cp.empty((nao_pair, naux_batch))
            err = kern(
                ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.c_int(SHM_SIZE), ctypes.c_int(naux_batch),
                ctypes.c_int(nbatches_shl_pair), ctypes.c_int(nblocks),
                ctypes.c_int(ksh_offsets[start]),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_batch.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(nst_lookup.data.ptr, ctypes.c_void_p),
                _atm_cpu.ctypes, ctypes.c_int(sorted_mol.natm),
                _bas_cpu.ctypes, ctypes.c_int(sorted_mol.nbas), _env_cpu.ctypes)
            if err != 0:
                raise RuntimeError('fill_int3c2e_bdiv kernel failed')
            if log.verbose >= logger.DEBUG1:
                cp.cuda.Stream.null.synchronize()
                log.timer_debug1('processing int3c2e_bdiv_kernel[:,{start}:{stop}]', *t0)
            yield eri3c
            eri3c = None

    def create_ao_pair_mapping(self, cart=True):
        '''ao_pair_mapping stores AO-pair addresses in the nao x nao matrix,
        which allows the decompression for the CUDA kernel generated compressed_eri3c:
        sparse_eri3c[ao_pair_mapping] = compressed_eri3c

        int3c2e CUDA kernel stores intgrals as [ij_shl,j,i,k,ksh].
        ao_pair_mapping indicates the ij addresses in eri3c[k,i,j];
        '''
        mol = self.sorted_mol
        ao_loc = cp.asarray(mol.ao_loc_nr(cart))
        nao = ao_loc[-1]
        uniq_l = self.uniq_l_ctr[:,0]
        if cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
        carts = [cp.arange(n) for n in nf]
        n_groups = len(uniq_l)
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        nbas = mol.nbas
        ao_pair_mapping = []
        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            ish, jsh = divmod(bas_ij_idx, nbas)
            iaddr = ao_loc[ish,None] + carts[i]
            jaddr = ao_loc[jsh,None] + carts[j]
            ao_pair_mapping.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
        return cp.hstack(ao_pair_mapping)

    def orbital_pair_cart2sph(self, compressed_eri3c, inplace=True):
        '''Transforms the AO of the compressed eri3c from Cartesian to spherical basis'''
        if inplace:
            out = compressed_eri3c
        else:
            out = compressed_eri3c.copy()
        uniq_l = self.uniq_l_ctr[:,0]
        n_groups = len(uniq_l)
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        c2s = [cart2sph_by_l(l) for l in uniq_l]
        naux = compressed_eri3c.shape[1]
        npair0 = npair = 0
        p0 = p1 = 0
        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            nshl_pair = bas_ij_idx.size
            ci = c2s[i]
            cj = c2s[j]
            nfi, di = ci.shape
            nfj, dj = cj.shape
            npair0, npair = npair, npair + nfi*nfj * nshl_pair
            p0, p1 = p1, p1 + di*dj * nshl_pair
            if npair0 > len(compressed_eri3c):
                raise RuntimeError('Size mismatch. The eri3c may have been transformed')
            t = compressed_eri3c[npair0:npair].reshape(nshl_pair,nfj,nfi,naux)
            t = contract('mpqr,pj->mjqr', t, cj)
            t = contract('mjqr,qi->mjir', t, ci)
            out[p0:p1] = t.reshape(p1-p0,naux)
        return out[:p1]

    def sort_orbitals(self, mat, axis=[]):
        ''' Transform given axis of a matrix into sorted AO'''
        ndim_to_transform = len(axis)
        assert ndim_to_transform <= 2
        if ndim_to_transform == 0:
            return mat

        idx = self.ao_idx
        fancy_index = [slice(None)] * mat.ndim
        if ndim_to_transform == 1:
            fancy_index[axis[0]] = idx
        elif ndim_to_transform == 2:
            assert abs(axis[0] - axis[1]) == 1, 'Must be adjacent axes'
            fancy_index[axis[0]] = idx[:,None]
            fancy_index[axis[1]] = idx
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[]):
        '''sort_orbitals reversed, transform the matrix in sorted AOs back to
        the original matrix.
        '''
        ndim_to_transform = len(axis)
        assert ndim_to_transform <= 2
        if ndim_to_transform == 0:
            return sorted_mat

        idx = self.ao_idx
        fancy_index = [slice(None)] * sorted_mat.ndim
        if ndim_to_transform == 1:
            fancy_index[axis[0]] = idx
        elif ndim_to_transform == 2:
            assert abs(axis[0] - axis[1]) == 1, 'Must be adjacent axes'
            fancy_index[axis[0]] = idx[:,None]
            fancy_index[axis[1]] = idx
        mat = cp.empty_like(sorted_mat)
        mat[tuple(fancy_index)] = sorted_mat
        return mat

    def orbital_pair_nonzero_indices(self):
        '''Returns rows, cols and diags, which are non-zero indices for orbital pairs.

        rows and cols are indices to address the elements in the (N,N) matrix for
        orbitals: ovlp[rows,cols] => non-zero. diags are the addresses in the
        compressed orbital pairs: ovlp[rows,cols][diag] => diagonal non-zero.
        diags contain the addresses of some of the off-diagonal elements
        '''
        mol = self.mol
        ao_pair_mapping = self.create_ao_pair_mapping(cart=mol.cart)
        rows, cols = divmod(asarray(ao_pair_mapping), mol.nao)

        # diag stores the indices for cderi_row that corresponds to
        # the diagonal blocks. Note this index array can contain some of the
        # off-diagonal elements which happen to be the off-diagonal elements
        # while within the diagonal blocks.
        uniq_l = self.uniq_l_ctr[:,0]
        if mol.cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
        n_groups = len(uniq_l)
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        nbas = self.sorted_mol.nbas
        offset = 0
        diag = []
        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            nfi = nf[i]
            nfj = nf[j]
            if i == j: # the diagonal blocks
                ish, jsh = divmod(bas_ij_idx, nbas)
                idx = np.where(ish == jsh)[0]
                addr = offset + idx[:,None] * (nfi*nfi) + np.arange(nfi*nfi)
                diag.append(addr.ravel())
            offset += bas_ij_idx.size * nfi * nfj
        diag = asarray(np.hstack(diag))
        return rows, cols, diag

def _conc_locs(ao_loc1, ao_loc2):
    return np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('log_cutoff', ctypes.c_float),
    ]

def init_constant(mol):
    g_idx, offsets = g_pair_idx()
    err = libgint_rys.init_constant(
        g_idx.ctypes, offsets.ctypes, mol._env.ctypes, ctypes.c_int(mol._env.size),
        ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

def int3c2e_scheme(li, lj, lk, shm_size=SHM_SIZE):
    order = li + lj + lk
    nroots = (order//2 + 1) * 2

    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3 + nroots*2 + 7
    nst_max = shm_size//(unit*8)
    nst_max = _nearest_power2(nst_max)

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    gout_size = nfi * nfj * nfk
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    gout_stride = min(gout_stride, 64)

    nst_per_block = min(nst_max, THREADS // gout_stride)
    gout_stride = THREADS // nst_per_block
    return nst_per_block, gout_stride

def _estimate_shl_pairs_per_block(li, lj, nshl_pair):
    return _nearest_power2(THREADS*2 // ((li+1)*(lj+1)), return_leq=False)

def create_nst_lookup_table():
    nst_lookup = np.empty([L_AUX_MAX+1]*3, dtype=np.int32)
    for lk in range(L_AUX_MAX+1):
        for li in range(lk+1):
            for lj in range(li+1):
                nst_lookup[lk,li,lj] = int3c2e_scheme(li, lj, lk)[0]
    idx = np.arange(L_AUX_MAX+1)
    z, y, x = np.sort(np.meshgrid(idx, idx, idx), axis=0)
    nst_lookup = nst_lookup[x, y, z]
    return nst_lookup[:,:LMAX+1,:LMAX+1]

def estimate_shl_ovlp(mol):
    # consider only the most diffused component of a basis
    exps, cs = extract_pgto_params(mol, 'diffused')
    exps = cp.asarray(exps)
    cs = cp.asarray(cs)
    ls = cp.asarray(mol._bas[:,ANG_OF])
    bas_coords = cp.asarray(mol.atom_coords()[mol._bas[:,ATOM_OF]])

    aij = exps[:,None] + exps
    fi = exps[:,None] / aij
    fj = exps[None,:] / aij
    theta = exps[:,None] * fj

    dr = dist_matrix(bas_coords, bas_coords)
    dri = fj * dr
    drj = fi * dr
    li = ls[:,None]
    lj = ls[None,:]
    fac_dri = (li * .5/aij + dri**2) ** (li*.5)
    fac_drj = (lj * .5/aij + drj**2) ** (lj*.5)
    fac_norm = cs[:,None]*cs * (np.pi/aij)**1.5
    ovlp = fac_norm * cp.exp(-theta*dr**2) * fac_dri * fac_drj
    return ovlp

def group_blocks(offsets, block_size):
    '''Partition shells into groups. num functions in each group <= block_size'''
    offsets = np.asarray(offsets)
    nbas = len(offsets) - 1
    partitions = []
    i = 0
    while i < nbas:
        partitions.append(i)
        upper_lim = offsets[i] + block_size
        next_i = np.searchsorted(offsets[i:], upper_lim, 'right')
        if next_i == 1:
            dim_max = (offsets[1:] - offsets[:-1]).max()
            raise RuntimeError(f'block_size {block_size} is too small. '
                               f'block_size should be at least {dim_max}.')
        i += next_i - 1
    partitions.append(min(i, nbas))
    return partitions
