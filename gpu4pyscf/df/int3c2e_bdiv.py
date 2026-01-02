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
from pyscf import gto
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, conc_env
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, dist_matrix, asarray, hermi_triu, transpose_sum,
    ndarray, get_avail_mem)
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.gto.mole import basis_seg_contraction, extract_pgto_params, cart2sph_by_l
from gpu4pyscf.gto.mole import SortedMole, RysIntEnvVars
from gpu4pyscf.scf.jk import (
    g_pair_idx, _nearest_power2, _scale_sp_ctr_coeff, _create_q_cond, SHM_SIZE,
    libvhf_rys)

__all__ = [
    'aux_e2',
]
libgint_rys = load_library('libgint_rys')
libgint_rys.fill_int3c2e.restype = ctypes.c_int
libgint_rys.fill_int3c2e_bdiv.restype = ctypes.c_int
libgint_rys.init_constant.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 54
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

def contract_int3c2e_dm(mol, auxmol, dm):
    int3c2e_opt = Int3c2eOpt_v2(mol, auxmol).build()
    dm = int3c2e_opt.mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm)
    return int3c2e_opt.auxmol.CT_dot_mat(auxvec)

def contract_int3c2e_auxvec(mol, auxmol, auxvec):
    int3c2e_opt = Int3c2eOpt_v2(mol, auxmol).build()
    auxvec = int3c2e_opt.auxmol.C_dot_mat(auxvec)
    vj = int3c2e_opt.contract_auxvec(auxvec)
    return int3c2e_opt.mol.apply_CT_mat_C(vj)

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

        auxmol, aux_idx, l_ctr_aux_pad_counts, uniq_l_ctr_aux, l_ctr_aux_counts = group_basis(
            self.auxmol, tile=1, sparse_coeff=True)
        self.sorted_auxmol = auxmol
        self.uniq_l_ctr_aux = uniq_l_ctr_aux
        l_ctr_aux_offsets = self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_aux_counts))
        self.l_ctr_aux_pad_counts = l_ctr_aux_pad_counts
        self.aux_idx = aux_idx

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
        self.int3c2e_envs = Int3c2eEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc, math.log(cutoff))

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
            t_ij = (cp.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    cp.arange(jsh0, jsh1, dtype=np.int32))
            idx = t_ij[mask[ish0:ish1,jsh0:jsh1]]
            nshl_pair = idx.size
            shl_pair_idx.append(idx)
            nfi = (li + 1) * (li + 2) // 2
            nfj = (lj + 1) * (lj + 2) // 2
            nfij = nfi * nfj
            nao_pair0, nao_pair = nao_pair, nao_pair + nfij * nshl_pair

            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = THREADS*2
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
        omega = mol.omega
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
                scheme = int3c2e_scheme(li, lj, lk, omega)
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
        omega = self.sorted_mol.omega
        nst_lookup, shm_size = create_nst_lookup_table(omega)
        nst_lookup = cp.asarray(nst_lookup, dtype=np.int32)
        max_shm_size = shm_size.max()

        shl_pair_idx = asarray(np.hstack(self.shl_pair_idx), dtype=np.int32)
        shl_pair_offsets = asarray(self.shl_pair_offsets, dtype=np.int32)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        ao_pair_loc = asarray(self.ao_pair_loc, dtype=np.int32)

        if batch_size is None:
            ksh_offsets = self.ksh_offsets
            ksh_offsets_gpu = cp.asarray(ksh_offsets, dtype=np.int32)
            ksh_blocks = len(ksh_offsets_gpu) - 1
            ksh_block_partitions = [0, ksh_blocks]
            aux_loc_by_block = aux_loc[ksh_offsets-sorted_mol.nbas]
        else:
            l_ctr_aux_offsets = self.l_ctr_aux_offsets
            uniq_l_ctr_aux = self.uniq_l_ctr_aux
            ksh_offsets = _split_l_ctr_pattern(l_ctr_aux_offsets, uniq_l_ctr_aux, batch_size)[0]
            aux_loc_by_block = aux_loc[ksh_offsets]
            ksh_offsets = ksh_offsets + sorted_mol.nbas
            ksh_offsets_gpu = cp.asarray(ksh_offsets, dtype=np.int32)
            ksh_blocks = len(ksh_offsets_gpu) - 1
            ksh_block_partitions = range(0, ksh_blocks+1)

        log.debug1('sp_blocks = %d, ksh_blocks = %d', nbatches_shl_pair, ksh_blocks)

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
                ctypes.c_int(max_shm_size), ctypes.c_int(naux_batch),
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
            if nshl_pair == 0:
                continue
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
                idx = cp.where(ish == jsh)[0].get()
                addr = offset + idx[:,None] * (nfi*nfi) + np.arange(nfi*nfi)
                diag.append(addr.ravel())
            offset += bas_ij_idx.size * nfi * nfj
        diag = asarray(np.hstack(diag))
        return rows, cols, diag

    @property
    def aux_coeff(self):
        coeff = np.zeros((self.sorted_auxmol.nao, self.auxmol.nao))

        l_max = max([l_ctr[0] for l_ctr in self.uniq_l_ctr_aux])
        if self.mol.cart:
            cart2sph_per_l = [np.eye((l+1)*(l+2)//2) for l in range(l_max + 1)]
        else:
            cart2sph_per_l = [gto.mole.cart2sph(l, normalized = "sp") for l in range(l_max + 1)]
        i_spherical_offset = 0
        i_cartesian_offset = 0
        for i, l in enumerate(self.uniq_l_ctr_aux[:,0]):
            cart2sph = cart2sph_per_l[l]
            ncart, nsph = cart2sph.shape
            l_ctr_count = self.l_ctr_aux_offsets[i + 1] - self.l_ctr_aux_offsets[i]
            cart_offs = i_cartesian_offset + np.arange(l_ctr_count) * ncart
            sph_offs = i_spherical_offset + np.arange(l_ctr_count) * nsph
            cart_idx = cart_offs[:,None] + np.arange(ncart)
            sph_idx = sph_offs[:,None] + np.arange(nsph)
            coeff[cart_idx[:,:,None],sph_idx[:,None,:]] = cart2sph
            l_ctr_pad_count = self.l_ctr_aux_pad_counts[i]
            i_cartesian_offset += (l_ctr_count + l_ctr_pad_count) * ncart
            i_spherical_offset += l_ctr_count * nsph
        out = np.empty_like(coeff)
        out[:,self.aux_idx] = coeff
        return out

class Int3c2eOpt_v2:
    def __init__(self, mol, auxmol):
        self.mol = SortedMole.from_mol(
            mol, allow_replica=True, allow_split_seg_contraction=False)
        self.auxmol = SortedMole.from_mol(
            auxmol, allow_replica=True, allow_split_seg_contraction=False)
        self._int3c2e_envs = None
        self.bas_ij_cache = None

    def build(self, cutoff=1e-14):
        mol = self.mol
        auxmol = self.auxmol
        assert all(self.mol.recontract_coef == 1.), \
                'int3c2e for general-contraction basis not supported'
        _atm, _bas, _env = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas[mol.nbas:,PTR_BAS_COORD] += off
        ao_loc = mol.ao_loc
        aux_loc = auxmol.ao_loc
        ao_loc = cp.asarray(_conc_locs(ao_loc, aux_loc), dtype=np.int32)
        self._int3c2e_envs = RysIntEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)
        l_ctr_offsets = np.append(0, np.cumsum(mol.l_ctr_counts))
        q_cond = _create_q_cond(mol, mol.uniq_l_ctr, l_ctr_offsets,
                                self._int3c2e_envs, cutoff)[0]
        mask = q_cond > math.log(cutoff)
        self.bas_ij_cache = mol.generate_shl_pairs(mask=mask)
        return self

    @property
    def int3c2e_envs(self):
        _int3c2e_envs = self._int3c2e_envs
        if _int3c2e_envs is None or cp.cuda.device.get_device_id() == _int3c2e_envs.device:
            return self._int3c2e_envs
        return _int3c2e_envs.copy()

    def _pair_and_diag_indices(self):
        mol = self.mol
        nbas = mol.nbas
        ao_loc = cp.asarray(mol.ao_loc)
        nao = ao_loc[-1]
        uniq_l = mol.uniq_l_ctr[:,0]
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
        carts = [cp.arange(n) for n in nf]
        # diag stores the indices for cderi_row that corresponds to
        # the diagonal blocks. Note this index array can contain some of the
        # off-diagonal elements which happen to be the off-diagonal elements
        # while within the diagonal blocks.
        offset = 0
        diag = []
        ao_pair_addresses = []
        for (i, j), bas_ij in self.bas_ij_cache.items():
            ish, jsh = divmod(bas_ij, nbas)
            iaddr = ao_loc[ish,None] + carts[i]
            jaddr = ao_loc[jsh,None] + carts[j]
            ao_pair_addresses.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
            if i == j: # the diagonal blocks
                nfi = nf[i]
                idx = cp.where(ish == jsh)[0]
                addr = offset + idx[:,None] * (nfi*nfi) + cp.arange(nfi*nfi)
                diag.append(addr.ravel())
            offset += len(bas_ij) * nf[i] * nf[j]
        ao_pair_addresses = cp.hstack(ao_pair_addresses)
        diag = cp.hstack(diag)
        return ao_pair_addresses, diag

    def int3c2e_evaluator(self, ao_pair_batch_size=None, aux_batch_size=None,
                          reorder_aux=False):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        mol = self.mol
        auxmol = self.auxmol
        nsp_per_block, gout_stride, shm_size = _int3c2e_scheme(mol.omega, gout_width=54)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block[0]*4)
        ao_pair_loc = get_ao_pair_loc(mol, self.bas_ij_cache)
        aux_loc = auxmol.ao_loc

        if ao_pair_batch_size is None:
            pair_splits = [0, len(shl_pair_offsets)-1]
            ao_pair_offsets = [0, ao_pair_loc[-1].get()]
        else:
            ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
            pair_splits = splits_by_blocksize(ao_pair_offsets, ao_pair_batch_size)
            ao_pair_offsets = ao_pair_offsets[pair_splits]

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
        uniq_l_ctr_aux = auxmol.uniq_l_ctr
        if aux_batch_size is None:
            ksh_offsets_cpu = l_ctr_aux_offsets
            ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
            aux_splits = [0, len(ksh_offsets_cpu)-1]
            aux_offsets = [0, aux_loc[-1]]
        else:
            l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
                l_ctr_aux_offsets, uniq_l_ctr_aux, aux_batch_size)
            ksh_offsets_cpu = l_ctr_aux_offsets
            ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
            aux_splits = range(len(ksh_offsets_cpu))
            aux_offsets = aux_loc[ksh_offsets_cpu]
        if reorder_aux:
            aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
        else:
            aux_sorting = np.arange(aux_offsets[-1])

        shl_pair_blocks = len(pair_splits) - 1
        ksh_blocks = len(ksh_offsets_cpu) - 1
        log.debug1('sp_blocks = %d, ksh_blocks = %d', shl_pair_blocks, ksh_blocks)

        kern = libvhf_rys.fill_int3c2e
        int3c2e_envs = self.int3c2e_envs

        def evaluate_j3c(shl_pair_batch_id=0, aux_batch_id=0, out=None):
            pair_split0 = pair_splits[shl_pair_batch_id]
            pair_split1 = pair_splits[shl_pair_batch_id+1]
            ao_pair_offset = ao_pair_offsets[shl_pair_batch_id]
            nao_pair = ao_pair_offsets[shl_pair_batch_id+1] - ao_pair_offset

            aux_split0 = aux_splits[aux_batch_id]
            aux_split1 = aux_splits[aux_batch_id+1]
            aux_ao_offset = aux_offsets[aux_batch_id]
            naux = aux_offsets[aux_batch_id+1] - aux_ao_offset
            out = ndarray((nao_pair, naux), buffer=out)
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(pair_split1 - pair_split0),
                ctypes.c_int(aux_split1 - aux_split0),
                ctypes.cast(shl_pair_offsets[pair_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ao_pair_offset), ctypes.c_int(aux_ao_offset),
                ctypes.c_int(naux), ctypes.c_int(reorder_aux))
            if err != 0:
                raise RuntimeError('fill_int3c2e kernel failed')
            return out
        return evaluate_j3c, ao_pair_offsets, aux_offsets, aux_sorting

    def int3c2e_bdiv_generator(self, cutoff=1e-14, batch_size=None, verbose=None):
        '''An iterator to generate eri3c blocks using the block-divergent
        integral parallelism
        '''
        # warning('Deprecated')
        evaluate_j3c, _, aux_offsets = self.int3c2e_evaluator(
            aux_batch_size=batch_size)[:3]
        for batch_id in range(len(aux_offsets)-1):
            yield evaluate_j3c(aux_batch_id=batch_id).T

    def contract_dm(self, dm, hermi=0):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        assert dm.ndim == 2
        assert dm.shape[1] == mol.nao
        assert dm.dtype == np.float64
        assert dm.flags.c_contiguous
        nbas_aux = auxmol.nbas
        if hermi != 1:
            dm = transpose_sum(dm, inplace=False)

        nsp_per_block, gout_stride, shm_size = _int3c2e_scheme(mol.omega)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block[0]*16)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        int3c2e_envs = self.int3c2e_envs
        naux = auxmol.nao
        vj_aux = cp.zeros(naux)
        err = libvhf_rys.contract_int3c2e_dm(
            ctypes.cast(vj_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(nbas_aux),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm failed')
        if hermi == 1:
            vj_aux *= 2
        log.timer_debug1('processing contract_int3c2e_dm', *t0)
        return vj_aux

    def contract_auxvec(self, auxvec):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        assert auxvec.ndim == 1
        auxvec = cp.asarray(auxvec)

        nsp_per_block, gout_stride, shm_size = _int3c2e_scheme(mol.omega, gout_width=30)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block[0]*4)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
        ksh_offsets = cp.asarray(l_ctr_aux_offsets + mol.nbas, dtype=np.int32)
        log.debug1('sp_blocks = %d, ksh_blocks = %d, shm_size = %d B',
                   len(shl_pair_offsets)-1, len(ksh_offsets)-1, shm_size_max)

        int3c2e_envs = self.int3c2e_envs
        nao = mol.nao
        vj = cp.zeros((nao, nao))
        err = libvhf_rys.contract_int3c2e_auxvec(
            ctypes.cast(vj.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(len(ksh_offsets) - 1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('contract_int3c2e_auxvec kernel failed')
        log.timer_debug1('processing contract_int3c2e_auxvec', *t0)
        vj = hermi_triu(vj, inplace=True)
        return vj

    def orbital_pair_cart2sph(self, compressed_eri3c, inplace=True):
        '''Transforms the AO of the compressed eri3c from Cartesian to spherical basis'''
        if self.bas_ij_cache is None:
            self.build()
        if inplace:
            out = compressed_eri3c
        else:
            out = compressed_eri3c.copy()
        uniq_l = self.mol.uniq_l_ctr[:,0]
        c2s = [cart2sph_by_l(l) for l in uniq_l]
        naux = compressed_eri3c.shape[1]
        npair0 = npair = 0
        p0 = p1 = 0
        for (i, j), bas_ij_idx in self.bas_ij_cache.items():
            nshl_pair = bas_ij_idx.size
            if nshl_pair == 0:
                continue
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

def _conc_locs(ao_loc1, ao_loc2):
    return np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_int),
        ('nbas', ctypes.c_int),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('log_cutoff', ctypes.c_float),
    ]

    @classmethod
    def new(cls, natm, nbas, atm, bas, env, ao_loc, log_cutoff):
        obj = Int3c2eEnvVars(natm, nbas, atm.data.ptr, bas.data.ptr,
                             env.data.ptr, ao_loc.data.ptr, log_cutoff)
        # Keep a reference to these arrays, prevent releasing them upon returning
        obj._env_ref_holder = (atm, bas, env, ao_loc)
        obj._device = cp.cuda.device.get_device_id()
        return obj

    def copy(self):
        atm, bas, env, ao_loc = self._env_ref_holder
        atm = cp.asarray(atm)
        bas = cp.asarray(bas)
        env = cp.asarray(env)
        ao_loc = cp.asarray(ao_loc)
        return Int3c2eEnvVars.new(self.natm, self.nbas, atm, bas, env, ao_loc,
                                  self.log_cutoff)

def init_constant(mol):
    g_idx, offsets = g_pair_idx()
    err = libgint_rys.init_constant(
        g_idx.ctypes, offsets.ctypes, mol._env.ctypes, ctypes.c_int(mol._env.size),
        ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

def int3c2e_scheme(li, lj, lk, omega=0, shm_size=SHM_SIZE):
    order = li + lj + lk
    nroots = order//2 + 1
    if omega < 0:
        nroots *= 2

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

def create_nst_lookup_table(omega=0):
    ls = np.arange(LMAX+1)
    li = ls[:,None]
    lj = ls
    laux = np.arange(L_AUX_MAX+1)[:,None,None]
    order = laux + li + lj
    nroots = order // 2 + 1
    if omega < 0:
        nroots *= 2
    g_size = (laux+1)*(li+1)*(lj+1)
    unit = g_size*3 + nroots*2 + 7
    shm_size = SHM_SIZE - 1024
    nst_max = shm_size // (unit*8)
    # Round to the nearest power2
    nst_max = _nearest_power2(nst_max)

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfaux = (laux + 1) * (laux + 2) // 2
    gout_size = nfaux * nfi * nfj
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    #gout_stride[gout_stride>=64] = 64

    nst_per_block = THREADS // gout_stride
    # min(nst_per_block, nst_max)
    nst_per_block = np.where(nst_per_block < nst_max, nst_per_block, nst_max)
    shm_size = nst_per_block * (unit * 8)
    shm_size += (nfaux + nfi + nfj) * (3 * 4)
    return nst_per_block, shm_size

def _int3c2e_scheme(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    order = li + lj + lk
    nroots = order//2 + 1
    if omega < 0:
        nroots *= 2 # for short-range
    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3 + nroots*2 + 7
    shm_size = shm_size - (nfi + nfj + nfk) * 3 * 4
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        gout_size = nfi * nfj * nfk
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    shm_size += (nfi + nfj + nfk) * 3 * 4
    return nsp_per_block, gout_stride, shm_size

def int2c2e_scheme(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(L_AUX_MAX+1)[:,None]
    lj = np.arange(L_AUX_MAX+1)
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    order = li + lj
    nroots = order//2 + 1
    if omega < 0:
        nroots *= 2 # for short-range
    g_size = (li+1)*(lj+1)
    unit = g_size*3 + nroots*2 + 4
    shm_size = shm_size - (nfi + nfj) * 3 * 4
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        gout_size = nfi * nfj
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    shm_size += (nfi + nfj) * 3 * 4
    return nsp_per_block, gout_stride, shm_size

def estimate_shl_ovlp(mol):
    # consider only the most diffuse component of a basis
    exps, cs = extract_pgto_params(mol, 'diffuse')
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

def _split_l_ctr_pattern(l_ctr_offsets, uniq_l_ctr, batch_size):
    '''
    Split l_ctr patterns into smaller chunks.
    '''
    l = uniq_l_ctr[:,0]
    nf = (l + 1) * (l + 2) // 2
    l_ctr_counts = l_ctr_offsets[1:] - l_ctr_offsets[:-1]
    if any(l_ctr_counts * nf > batch_size):
        counts = l_ctr_counts.tolist()
        repeats = []
        for i, count in enumerate(l_ctr_counts):
            mxshl_in_batch = max(batch_size // nf[i], 1)
            repeat, remainder = divmod(count, mxshl_in_batch)
            expand = [mxshl_in_batch] * repeat
            if remainder != 0:
                expand.append(remainder)
                repeat += 1
            counts[i] = expand
            repeats.append(repeat)
        l_ctr_counts = np.hstack(counts)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        uniq_l_ctr = np.repeat(uniq_l_ctr, repeats, axis=0)
    return l_ctr_offsets, uniq_l_ctr

def argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux):
    '''
    The auxiliary functions are sorted to
    [s,s,s,...,px,px,px,...,py,py,py,...,pz,pz,pz,...] than the
    conventional order [s,s,...,px,py,pz,px,py,pz,pz,...].
    aux_sorting maps the addresses of the two storge formats.
    '''
    l = uniq_l_ctr_aux[:,0]
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    aux_sorting = []
    nksh = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]
    for k, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux0, aux1 = aux1, aux1 + nf[k] * nksh[k]
        aux_sorting.append(cp.arange(aux0, aux1).reshape(nf[k], nksh[k]).T.ravel())
    return cp.hstack(aux_sorting)

def get_ao_pair_loc(mol, bas_ij_cache):
    '''
    For each primitive shell-pair in bas_ij_idx, ao_pair_loc points to the
    addresses of first element for the contracted pair-GTOs. In each
    shell-pair, there are nfij elements. Note, the nfij elements are
    sorted as [nfj,nfi] (in F-order).
    '''
    uniq_l = mol.uniq_l_ctr[:,0]
    nf = (uniq_l + 1) * (uniq_l + 2) // 2
    ao_pair_loc = []
    p0 = p1 = 0
    for (i, j), bas_ij in bas_ij_cache.items():
        nfij = nf[i] * nf[j]
        p0, p1 = p1, p1 + nfij * len(bas_ij)
        ao_pair_loc.append(cp.arange(p0, p1, nfij, dtype=np.int32))
    ao_pair_loc.append(np.int32(p1))
    ao_pair_loc = cp.asarray(cp.hstack(ao_pair_loc), dtype=np.int32)
    return ao_pair_loc

def _vector_sph2cart(auxmol, auxvec):
    aux_ls = auxmol._bas[:,ANG_OF]
    lmax = aux_ls.max()
    aux_loc_cart = auxmol.ao_loc_nr(cart=True)
    aux_loc_sph = auxmol.ao_loc_nr(cart=False)[:-1]
    naux_cart = aux_loc_cart[-1]
    aux_loc_cart = aux_loc_cart[:-1]
    out = cp.empty(naux_cart)
    for l in range(lmax+1):
        nf = (l + 1) * (l + 2) // 2
        addrs_for_cart = aux_loc_cart[aux_ls == l]
        addrs_for_sph = aux_loc_sph[aux_ls == l]
        subvec = auxvec[addrs_for_sph[:,None] + np.arange(l*2+1)]
        subvec = subvec.dot(cart2sph_by_l(l).T)
        out[addrs_for_cart[:,None] + np.arange(nf)] = subvec
    return out

def int2c2e(mol):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.gto.mole import PBCIntEnvVars
    from gpu4pyscf.pbc.df.ft_ao import libpbc
    libpbc.fill_int2c2e.restype = ctypes.c_int
    is_sorted_mol = isinstance(mol, SortedMole)
    if not is_sorted_mol:
        mol = SortedMole.from_mol(mol)
    lmax = mol.uniq_l_ctr[:,0].max()
    assert lmax <= L_AUX_MAX

    nsp_per_block, gout_stride, shm_size = int2c2e_scheme(mol.omega, gout_width=60)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    shm_size_max = shm_size[:lmax+1,:lmax+1].max()

    ao_loc = mol.ao_loc
    nao = ao_loc[-1]
    Ls = cp.zeros((1, 3))
    _env = _scale_sp_ctr_coeff(mol)
    rys_envs = PBCIntEnvVars.new(
        mol.natm, mol.nbas, 1, 1, mol._atm, mol._bas, _env, ao_loc, Ls)
    nbas = mol.nbas
    mask = cp.ones((nbas, nbas), dtype=bool)
    bas_ij_cache = mol.generate_shl_pairs(mask=mask)
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        bas_ij_cache, nsp_per_block)

    nbatches_shl_pair = len(shl_pair_offsets) - 1
    out = cp.empty((nao, nao))
    err = libpbc.fill_int2c2e(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('fill_int2c2e failed')
    out = hermi_triu(out, inplace=True)
    if not is_sorted_mol:
        out = mol.apply_CT_mat_C(out)
    return out

# TODO: move to pbc.gto.int1e module
def int2c2e_ip1(mol):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.gto.mole import PBCIntEnvVars
    from gpu4pyscf.pbc.df.ft_ao import libpbc
    libpbc.fill_int2c2e.restype = ctypes.c_int
    is_sorted_mol = isinstance(mol, SortedMole)
    if not is_sorted_mol:
        mol = SortedMole.from_mol(mol)
    lmax = mol.uniq_l_ctr[:,0].max()
    assert lmax <= L_AUX_MAX

    shm_size = SHM_SIZE
    li = np.arange(L_AUX_MAX+1)[:,None]
    lj = np.arange(L_AUX_MAX+1)
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    order = li + lj + 1
    nroots = order//2 + 1
    if mol.omega < 0:
        nroots *= 2 # for short-range
    g_size = (li+2)*(lj+1)
    unit = g_size*3 + nroots*2 + 4
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    gout_width = 20
    if gout_width is not None:
        gout_size = nfi * nfj
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    shm_size_max = shm_size[:lmax+1,:lmax+1].max()

    ao_loc = mol.ao_loc
    nao = ao_loc[-1]
    Ls = cp.zeros((1, 3))
    _env = _scale_sp_ctr_coeff(mol)
    rys_envs = PBCIntEnvVars.new(
        mol.natm, mol.nbas, 1, 1, mol._atm, mol._bas, _env, ao_loc, Ls)
    nbas = mol.nbas
    mask = cp.ones((nbas, nbas), dtype=bool)
    bas_ij_cache = mol.generate_shl_pairs(mask=mask, hermi=0)
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        bas_ij_cache, nsp_per_block)

    nbatches_shl_pair = len(shl_pair_offsets) - 1
    out = cp.empty((3, nao, nao))
    err = libpbc.fill_int2c2e_ip1(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('fill_int2c2e failed')
    if not is_sorted_mol:
        out = mol.apply_CT_mat_C(out)
    return out
