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

'''
Compute J/K matrices
'''

import ctypes
import warnings
import math
import numpy as np
import cupy as cp
from collections import Counter
from pyscf.gto import ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF
from pyscf import lib, gto
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import (
    load_library, condense, transpose_sum, reduce_to_device, hermi_triu,
    asarray, dist_matrix)
from gpu4pyscf.__config__ import num_devices, shm_size
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import group_basis, cart2sph_by_l, extract_pgto_params

__all__ = [
    'get_jk', 'get_j', 'get_k',
]

libvhf_rys = load_library('libgvhf_rys')
libvhf_rys.RYS_build_jk.restype = ctypes.c_int
libvhf_rys.RYS_init_constant.restype = ctypes.c_int
libvhf_rys.RYS_build_k.restype = ctypes.c_int
libvhf_rys.cuda_version.restype = ctypes.c_int
CUDA_VERSION = libvhf_rys.cuda_version()
libgint = load_library('libgint')

PTR_BAS_COORD = 7
LMAX = 4
TILE = 1
QUEUE_DEPTH = 262144
SHM_SIZE = shm_size - 1024
del shm_size
GOUT_WIDTH = 42
THREADS = 256
GROUP_SIZE = 256

libvhf_rys.RYS_build_k_init(ctypes.c_int(SHM_SIZE))
libvhf_rys.RYS_build_jk_init(ctypes.c_int(SHM_SIZE))

def get_jk(mol, dm, hermi=0, vhfopt=None, with_j=True, with_k=True, verbose=None):
    '''Compute J, K matrices
    '''
    assert with_j or with_k
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = _VHFOpt(mol, tile=1).build()
    assert vhfopt.tile == 1

    mol = vhfopt.sorted_mol
    nao_orig = vhfopt.mol.nao

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    dms = cp.asarray(dms, order='C')

    vj, vk = vhfopt.get_jk(dms, hermi, log)
    if with_k:
        #:vk = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vk, vhfopt.coeff)
        vk = vhfopt.apply_coeff_CT_mat_C(vk)
        vk = vk.reshape(dm.shape)
    else:
        vk = None

    if with_j:
        #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vj, vhfopt.coeff)
        vj = vhfopt.apply_coeff_CT_mat_C(vj)
        vj = vj.reshape(dm.shape)
    else:
        vj = None
    log.timer('vj and vk', *cput0)
    return vj, vk

def get_k(mol, dm, hermi=0, vhfopt=None, verbose=None):
    '''Compute K matrix
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = _VHFOpt(mol, tile=1).build()
    assert vhfopt.tile == 1

    mol = vhfopt.sorted_mol
    nao_orig = vhfopt.mol.nao

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    dms = cp.asarray(dms, order='C')

    vk = vhfopt.get_k(dms, hermi, log)
    #:vk = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vk, vhfopt.coeff)
    vk = vhfopt.apply_coeff_CT_mat_C(vk)
    vk = vk.reshape(dm.shape)
    log.timer('vk', *cput0)
    return vk

def get_j(mol, dm, hermi=0, vhfopt=None, verbose=None):
    '''Compute J matrix
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()
    assert vhfopt.tile == TILE

    nao_orig = vhfopt.mol.nao

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    assert n_dm == 1
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    dms = cp.asarray(dms, order='C')
    if hermi != 1:
        dms = transpose_sum(dms)
        dms *= .5

    vj = vhfopt.get_j(dms, log)
    #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, cp.asarray(vj), vhfopt.coeff)
    vj = vhfopt.apply_coeff_CT_mat_C(cp.asarray(vj))
    vj = vj.reshape(dm.shape)
    log.timer('vj', *cput0)
    return vj

def apply_coeff_C_mat_CT(spherical_matrix, mol, sorted_mol, uniq_l_ctr,
                         l_ctr_offsets, ao_idx, l_ctr_paddings=None):
    '''
    Unsort AO and perform sph2cart transformation (if needed) for the last 2 axes
    Fused kernel to perform 'ip,npq,qj->nij'
    '''
    spherical_matrix = cp.asarray(spherical_matrix)
    spherical_matrix_ndim = spherical_matrix.ndim
    if spherical_matrix_ndim == 2:
        spherical_matrix = spherical_matrix[None]
    n_spherical = mol.nao
    assert spherical_matrix.shape[1] == n_spherical
    assert spherical_matrix.shape[2] == n_spherical
    n_cartesian = sorted_mol.nao

    output_complex = False
    if spherical_matrix.dtype == np.complex128:
        spherical_matrix = spherical_matrix.view(np.float64)
        spherical_matrix = spherical_matrix.reshape(-1,n_spherical,n_spherical,2)
        spherical_matrix = spherical_matrix.transpose(3,0,1,2).reshape(-1,n_spherical,n_spherical)
        output_complex = True
    else:
        assert spherical_matrix.dtype == np.float64
    counts = spherical_matrix.shape[0]

    l_ctr_count = np.asarray(l_ctr_offsets[1:] - l_ctr_offsets[:-1], dtype = np.int32)
    l_ctr_l = np.asarray(uniq_l_ctr[:,0], dtype=np.int32, order='C')
    if l_ctr_paddings is None:
        l_ctr_pad_counts = np.zeros_like(l_ctr_count)
    else:
        l_ctr_pad_counts = np.asarray(l_ctr_paddings, dtype=np.int32)
    ao_idx = cp.asarray(ao_idx, dtype=np.int32)
    stream = cp.cuda.get_current_stream()

    out = cp.zeros((counts, n_cartesian, n_cartesian), order = "C")
    for i_dm in range(counts):
        libgint.cart2sph_C_mat_CT_with_padding(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(out[i_dm].data.ptr, ctypes.c_void_p),
            ctypes.cast(spherical_matrix[i_dm].data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_cartesian),
            ctypes.c_int(n_spherical),
            ctypes.c_int(l_ctr_l.shape[0]),
            l_ctr_l.ctypes.data_as(ctypes.c_void_p),
            l_ctr_count.ctypes.data_as(ctypes.c_void_p),
            l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(ao_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_bool(mol.cart),
        )

    if output_complex:
        outR, outI = out.reshape(2, -1, n_cartesian, n_cartesian)
        out = outR.astype(np.complex128)
        out.imag = outI

    if spherical_matrix_ndim == 2:
        out = out[0]
    return out

def apply_coeff_CT_mat_C(cartesian_matrix, mol, sorted_mol, uniq_l_ctr,
                         l_ctr_offsets, ao_idx, l_ctr_paddings=None):
    '''
    Sort AO and perform cart2sph transformation (if needed) for the last 2 axes
    Fused kernel to perform 'ip,npq,qj->nij'
    '''
    cartesian_matrix = cp.asarray(cartesian_matrix)
    cartesian_matrix_ndim = cartesian_matrix.ndim
    if cartesian_matrix_ndim == 2:
        cartesian_matrix = cartesian_matrix[None]
    n_cartesian = sorted_mol.nao
    assert cartesian_matrix.shape[1] == n_cartesian
    assert cartesian_matrix.shape[2] == n_cartesian
    n_spherical = mol.nao

    output_complex = False
    if cartesian_matrix.dtype == np.complex128:
        cartesian_matrix = cartesian_matrix.view(np.float64)
        cartesian_matrix = cartesian_matrix.reshape(-1,n_cartesian,n_cartesian,2)
        cartesian_matrix = cartesian_matrix.transpose(3,0,1,2).reshape(-1,n_cartesian,n_cartesian)
        output_complex = True
    else:
        assert cartesian_matrix.dtype == np.float64
    counts = cartesian_matrix.shape[0]

    l_ctr_count = np.asarray(l_ctr_offsets[1:] - l_ctr_offsets[:-1], dtype = np.int32)
    l_ctr_l = np.asarray(uniq_l_ctr[:,0], dtype=np.int32, order='C')
    if l_ctr_paddings is None:
        l_ctr_pad_counts = np.zeros_like(l_ctr_count)
    else:
        l_ctr_pad_counts = np.asarray(l_ctr_paddings, dtype=np.int32)
    ao_idx = cp.asarray(ao_idx, dtype=np.int32)
    stream = cp.cuda.get_current_stream()

    out = cp.empty((counts, n_spherical, n_spherical), order = "C")
    for i_dm in range(counts):
        libgint.cart2sph_CT_mat_C_with_padding(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(cartesian_matrix[i_dm].data.ptr, ctypes.c_void_p),
            ctypes.cast(out[i_dm].data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_cartesian),
            ctypes.c_int(n_spherical),
            ctypes.c_int(l_ctr_l.shape[0]),
            l_ctr_l.ctypes.data_as(ctypes.c_void_p),
            l_ctr_count.ctypes.data_as(ctypes.c_void_p),
            l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(ao_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_bool(mol.cart),
        )

    if output_complex:
        outR, outI = out.reshape(2, -1, n_spherical, n_spherical)
        out = outR.astype(np.complex128)
        out.imag = outI

    if cartesian_matrix_ndim == 2:
        out = out[0]
    return out


class _VHFOpt:
    def __init__(self, mol, direct_scf_tol=1e-13, tile=TILE):
        self.mol = mol
        self.sorted_mol = None
        self.direct_scf_tol = direct_scf_tol
        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.h_shls = None
        self.tile = tile

        # Hold cache on GPU devices
        self._rys_envs = {}
        self._q_cond = {}
        self._tile_q_cond = {}
        self._s_estimator = {}
        self._cupy_ao_idx = {}

    def reset(self, mol):
        self.mol = mol
        self.sorted_mol = None
        self._rys_envs = {}
        self._q_cond = {}
        self._tile_q_cond = {}
        self._s_estimator = {}
        self._cupy_ao_idx = {}

    def build(self, group_size=None, verbose=None):
        mol = self.mol
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = group_basis(
            mol, self.tile, group_size, sparse_coeff = True)
        self.sorted_mol = mol
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = np.asarray(l_ctr_pad_counts, dtype=np.int32)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        # very high angular momentum basis are processed on CPU
        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        q_cond, s_estimator = _create_q_cond(
            mol, uniq_l_ctr, self.l_ctr_offsets, self.rys_envs,
            self.direct_scf_tol)
        self.q_cond_cpu = q_cond.get()

        if mol.omega < 0:
            self.s_estimator_cpu = s_estimator.get()
        log.timer('Initialize q_cond', *cput0)
        return self

    def sort_orbitals(self, mat, axis=[]):
        '''
        Transform given axis of a matrix into sorted AO
        '''
        idx = self.ao_idx
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
        '''
        Transform given axis of a matrix into sorted AO
        '''
        idx = self.ao_idx
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

    def apply_coeff_C_mat_CT(self, spherical_matrix):
        '''
        Unsort AO and perform sph2cart transformation (if needed) for the last 2 axes
        Fused kernel to perform 'ip,npq,qj->nij'
        '''
        return apply_coeff_C_mat_CT(
            spherical_matrix, self.mol, self.sorted_mol, self.uniq_l_ctr,
            self.l_ctr_offsets, self.cupy_ao_idx, self.l_ctr_pad_counts)

    def apply_coeff_CT_mat_C(self, cartesian_matrix):
        '''
        Sort AO and perform cart2sph transformation (if needed) for the last 2 axes
        Fused kernel to perform 'ip,npq,qj->nij'
        '''
        return apply_coeff_CT_mat_C(
            cartesian_matrix, self.mol, self.sorted_mol, self.uniq_l_ctr,
            self.l_ctr_offsets, self.cupy_ao_idx, self.l_ctr_pad_counts)

    def apply_coeff_C_mat(self, right_matrix):
        '''
        Sort AO and perform cart2sph transformation (if needed) for the second last axis
        Fused kernel to perform 'ip,npq->niq'
        '''
        right_matrix = cp.asarray(right_matrix)
        assert right_matrix.ndim == 2
        assert right_matrix.shape[0] == self.mol.nao
        n_cartesian = self.sorted_mol.nao
        n_second = right_matrix.shape[1]

        l_ctr_count = np.asarray(self.l_ctr_offsets[1:] - self.l_ctr_offsets[:-1], dtype = np.int32)
        l_ctr_l = np.asarray(self.uniq_l_ctr[:,0].copy(), dtype = np.int32)
        self.l_ctr_pad_counts = np.asarray(self.l_ctr_pad_counts, dtype = np.int32)
        cupy_ao_idx = self.cupy_ao_idx
        stream = cp.cuda.get_current_stream()

        # ref = self.coeff @ right_matrix

        right_matrix = cp.ascontiguousarray(right_matrix)

        out = cp.zeros((n_cartesian, n_second), order = "C")
        libgint.cart2sph_C_mat_with_padding(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(right_matrix.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_second),
            ctypes.c_int(l_ctr_l.shape[0]),
            l_ctr_l.ctypes.data_as(ctypes.c_void_p),
            l_ctr_count.ctypes.data_as(ctypes.c_void_p),
            self.l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(cupy_ao_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_bool(self.mol.cart),
        )

        return out

    @multi_gpu.property(cache='_q_cond')
    def q_cond(self):
        return asarray(self.q_cond_cpu)

    @multi_gpu.property(cache='_s_estimator')
    def s_estimator(self):
        return asarray(self.s_estimator_cpu)

    @multi_gpu.property(cache='_cupy_ao_idx')
    def cupy_ao_idx(self):
        return asarray(self.ao_idx, dtype = cp.int32)

    @multi_gpu.property(cache='_rys_envs')
    def rys_envs(self):
        mol = self.sorted_mol
        _atm = cp.array(mol._atm)
        _bas = cp.array(mol._bas)
        _env = cp.array(_scale_sp_ctr_coeff(mol))
        ao_loc = cp.array(mol.ao_loc)
        return RysIntEnvVars.new(mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)

    @property
    def coeff(self):
        coeff = np.zeros((self.sorted_mol.nao, self.mol.nao))

        l_max = max([l_ctr[0] for l_ctr in self.uniq_l_ctr])
        if self.mol.cart:
            cart2sph_per_l = [np.eye((l+1)*(l+2)//2) for l in range(l_max + 1)]
        else:
            cart2sph_per_l = [gto.mole.cart2sph(l, normalized = "sp") for l in range(l_max + 1)]
        i_spherical_offset = 0
        i_cartesian_offset = 0
        for i, l in enumerate(self.uniq_l_ctr[:,0]):
            cart2sph = cart2sph_per_l[l]
            ncart, nsph = cart2sph.shape
            l_ctr_count = self.l_ctr_offsets[i + 1] - self.l_ctr_offsets[i]
            cart_offs = i_cartesian_offset + np.arange(l_ctr_count) * ncart
            sph_offs = i_spherical_offset + np.arange(l_ctr_count) * nsph
            cart_idx = cart_offs[:,None] + np.arange(ncart)
            sph_idx = sph_offs[:,None] + np.arange(nsph)
            coeff[cart_idx[:,:,None],sph_idx[:,None,:]] = cart2sph
            l_ctr_pad_count = self.l_ctr_pad_counts[i]
            i_cartesian_offset += (l_ctr_count + l_ctr_pad_count) * ncart
            i_spherical_offset += l_ctr_count * nsph
        assert len(self.ao_idx) == self.mol.nao
        coeff = self.unsort_orbitals(coeff, axis = [1])
        return asarray(coeff)

    def get_jk(self, dms, hermi, verbose):
        '''
        Build JK for the sorted_mol. Density matrices dms and the return JK
        matrices are all corresponding to the sorted_mol
        '''
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        log = logger.new_logger(mol, verbose)
        ao_loc = mol.ao_loc
        uniq_l_ctr = self.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = self.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        assert dms.ndim == 3 and dms.shape[-1] == ao_loc[-1]
        dm_cond = condense('absmax', dms, ao_loc)
        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.get_current_stream()
            log = logger.new_logger(mol, verbose)
            t0 = log.init_timer()
            dms = cp.asarray(dms) # transfer to current device
            dm_cond = cp.asarray(dm_cond)

            if hermi == 0:
                # Contract the tril and triu parts separately
                dms = cp.vstack([dms, dms.transpose(0,2,1)])
            n_dm, nao = dms.shape[:2]
            vj = cp.zeros(dms.shape)
            vk = cp.zeros(dms.shape)
            q_cond = cp.asarray(self.q_cond)
            s_ptr = lib.c_null_ptr()
            if mol.omega < 0:
                s_ptr = ctypes.cast(self.s_estimator.data.ptr, ctypes.c_void_p)
            pair_mappings = _make_tril_pair_mappings(
                l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
            rys_envs = self.rys_envs

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            # An additional integer to count for the proccessed pair_ijs
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)

            timing_counter = Counter()
            kern_counts = 0
            kern = libvhf_rys.RYS_build_jk

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping = pair_mappings[i,j]
                pair_kl_mapping = pair_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                    _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                    _npairs_kl = pair_kl1 - pair_kl0
                    err = kern(
                        ctypes.cast(vj.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                        ctypes.c_int(SHM_SIZE),
                        ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                        ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                        s_ptr,
                        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    if err != 0:
                        llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                        raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()

            if hermi == 1:
                vj *= 2.
                vk = transpose_sum(vk)
            else:
                vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
                vj += vjT.transpose(0,2,1)
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
            return vj, vk, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        if self.h_shls:
            dms = dms.get()
            dm_cond = None
        else:
            dms = dm_cond = None

        kern_counts = 0
        timing_collection = Counter()
        vj_dist = []
        vk_dist = []
        for vj, vk, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            vj_dist.append(vj)
            vk_dist.append(vk)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        vk = multi_gpu.array_reduce(vk_dist, inplace=True)
        vj = multi_gpu.array_reduce(vj_dist, inplace=True)
        vj = transpose_sum(vj)

        h_shls = self.h_shls
        if h_shls:
            log.debug3('Integrals for %s functions on CPU',
                       lib.param.ANGULAR[LMAX+1])
            scripts = ['ji->s2kl']
            if hermi == 1:
                scripts.append('jk->s2il')
            else:
                scripts.append('jk->s1il')
            shls_excludes = [0, h_shls[0]] * 4
            vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms, 1, mol._atm, mol._bas, mol._env,
                                     shls_excludes=shls_excludes)
            vj1 = asarray(vs_h[0])
            vk1 = asarray(vs_h[1])
            if hermi:
                vk1 = hermi_triu(vk1)
            vj += hermi_triu(vj1)
            vk += vk1
        return vj, vk

    def get_j(self, dms, verbose):
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        log = logger.new_logger(mol, verbose)
        ao_loc = mol.ao_loc
        n_dm, nao = dms.shape[:2]
        assert dms.ndim == 3 and nao == ao_loc[-1]
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        uniq_l_ctr = self.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = self.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        if isinstance(dms, cp.ndarray):
            dms = dms.get()
        pair_loc = _make_j_engine_pair_locs(mol)
        dm_xyz = np.empty(pair_loc[-1])
        libvhf_rys.transform_cart_to_xyz(
            dm_xyz.ctypes, dms.ctypes, ao_loc.ctypes, pair_loc.ctypes,
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)

        tasks = [(i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1)]
        schemes = {t: _j_engine_quartets_scheme(mol, uniq_l_ctr[list(t)]) for t in tasks}
        tasks = iter(tasks)
        libvhf_rys.RYS_init_rysj_constant.restype = ctypes.c_int

        def proc(dm_xyz, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.get_current_stream()
            log = logger.new_logger(mol, verbose)
            t0 = log.init_timer()
            dm_xyz = asarray(dm_xyz) # transfer to current device
            dm_cond = asarray(dm_cond)
            vj_xyz = cp.zeros_like(dm_xyz)
            pair_loc_on_gpu = asarray(pair_loc)
            _atm, _bas, _env, _ = self.rys_envs._env_ref_holder
            rys_envs = RysIntEnvVars(
                mol.natm, mol.nbas,
                _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
                pair_loc_on_gpu.data.ptr,
            )
            q_cond = self.q_cond

            err = libvhf_rys.RYS_init_rysj_constant(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError('CUDA kernel initialization')

            pair_mappings = _make_tril_pair_mappings(
                l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)

            timing_collection = {}
            kern_counts = 0
            try:
                kern = libvhf_rys.RYS_build_j
            except AttributeError:
                logger.error('RYS_build_j is not compiled')
                raise

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping = pair_mappings[i,j]
                pair_kl_mapping = pair_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = schemes[task]
                for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                    _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                    _npairs_kl = pair_kl1 - pair_kl0
                    err = kern(
                        ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        ctypes.byref(rys_envs), (ctypes.c_int*3)(*scheme),
                        (ctypes.c_int*8)(*shls_slice),
                        ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                        ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                        lib.c_null_ptr(),
                        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    if err != 0:
                        llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                        raise RuntimeError(f'RYS_build_j kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    if llll not in timing_collection:
                        timing_collection[llll] = 0
                    timing_collection[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dm_xyz, dm_cond), non_blocking=True)
        dm_xyz = dm_cond = None

        kern_counts = 0
        timing_collection = Counter()
        vj_dist = []
        for vj, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            vj_dist.append(vj)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        vj_xyz = multi_gpu.array_reduce(vj_dist, inplace=True)
        vj_xyz = vj_xyz.get()
        vj = np.empty_like(dms)
        libvhf_rys.transform_xyz_to_cart(
            vj.ctypes, vj_xyz.ctypes, ao_loc.ctypes, pair_loc.ctypes,
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
        vj = transpose_sum(asarray(vj))
        vj *= 2.

        h_shls = self.h_shls
        if h_shls:
            log.debug3('Integrals for %s functions on CPU',
                       lib.param.ANGULAR[LMAX+1])
            scripts = ['ji->s2kl']
            shls_excludes = [0, h_shls[0]] * 4
            vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms, 1, mol._atm, mol._bas, mol._env,
                                     shls_excludes=shls_excludes)
            vj1 = asarray(vs_h[0])
            vj += hermi_triu(vj1)
        return vj

    def get_k(self, dms, hermi, verbose):
        '''
        Build K matrix for the sorted_mol. Density matrices dms and the return K
        matrix are all corresponding to the sorted_mol
        '''
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        log = logger.new_logger(mol, verbose)
        ao_loc = mol.ao_loc
        uniq_l_ctr = self.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = self.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        assert dms.ndim == 3 and dms.shape[-1] == ao_loc[-1]
        dm_cond = condense('absmax', dms, ao_loc)
        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(mol, verbose)
            t0 = log.init_timer()
            dms = cp.asarray(dms) # transfer to current device
            dm_cond = cp.asarray(dm_cond)

            if hermi == 0:
                # Contract the tril and triu parts separately
                dms = cp.vstack([dms, dms.transpose(0,2,1)])
            n_dm, nao = dms.shape[:2]
            vk = cp.zeros(dms.shape)
            q_cond = cp.asarray(self.q_cond)
            s_ptr = lib.c_null_ptr()
            if mol.omega < 0:
                s_ptr = ctypes.cast(self.s_estimator.data.ptr, ctypes.c_void_p)
            pair_mappings = _make_tril_pair_mappings(
                l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
            rys_envs = self.rys_envs

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)

            timing_counter = Counter()
            kern_counts = 0
            kern = libvhf_rys.RYS_build_k

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping = pair_mappings[i,j]
                pair_kl_mapping = pair_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                    _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                    _npairs_kl = pair_kl1 - pair_kl0
                    err = kern(
                        ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                        ctypes.c_int(SHM_SIZE),
                        ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                        ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                        s_ptr,
                        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    if err != 0:
                        llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                        raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()

            if hermi == 1:
                vk = transpose_sum(vk)
            else:
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
            return vk, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        if self.h_shls:
            dms = dms.get()
            dm_cond = None
        else:
            dms = dm_cond = None

        kern_counts = 0
        timing_collection = Counter()
        vk_dist = []
        for vk, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            vk_dist.append(vk)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        vk = multi_gpu.array_reduce(vk_dist, inplace=True)

        h_shls = self.h_shls
        if h_shls:
            log.debug3('Integrals for %s functions on CPU',
                       lib.param.ANGULAR[LMAX+1])
            scripts = []
            if hermi == 1:
                scripts.append('jk->s2il')
            else:
                scripts.append('jk->s1il')
            shls_excludes = [0, h_shls[0]] * 4
            vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms, 1, mol._atm, mol._bas, mol._env,
                                     shls_excludes=shls_excludes)
            vk1 = asarray(vs_h[0])
            if hermi:
                vk1 = hermi_triu(vk1)
            vk += vk1
        return vk

class RysIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_int),
        ('nbas', ctypes.c_int),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
    ]

    @classmethod
    def new(cls, natm, nbas, atm, bas, env, ao_loc):
        obj = RysIntEnvVars(natm, nbas, atm.data.ptr, bas.data.ptr,
                            env.data.ptr, ao_loc.data.ptr)
        # Keep a reference to these arrays, prevent releasing them upon returning
        obj._env_ref_holder = (atm, bas, env, ao_loc)
        return obj

    def copy(self):
        atm, bas, env, ao_loc = self._env_ref_holder
        atm = cp.asarray(atm)
        bas = cp.asarray(bas)
        env = cp.asarray(env)
        ao_loc = cp.asarray(ao_loc)
        return RysIntEnvVars.new(self.natm, self.nbas, atm, bas, env, ao_loc)

    @property
    def device(self):
        return self._env_ref_holder[2].device

def _scale_sp_ctr_coeff(mol):
    # Match normalization factors of s, p functions in libcint
    _env = mol._env.copy()
    ls = mol._bas[:,ANG_OF]
    ptr, idx = np.unique(mol._bas[:,PTR_COEFF], return_index=True)
    ptr = ptr[ls[idx] < 2]
    idx = idx[ls[idx] < 2]
    fac = ((ls[idx]*2+1) / (4*np.pi)) ** .5
    nprim = mol._bas[idx,NPRIM_OF]
    nctr = mol._bas[idx,NCTR_OF]
    for p, n, f in zip(ptr, nprim*nctr, fac):
        _env[p:p+n] *= f
    return _env

def iter_cart_xyz(n):
    return [(x, y, n-x-y)
            for x in reversed(range(n+1))
            for y in reversed(range(n+1-x))]

def g_pair_idx(ij_inc=None):
    dat = []
    xyz = [np.array(iter_cart_xyz(li)) for li in range(LMAX+1)]
    for li in range(LMAX+1):
        for lj in range(LMAX+1):
            li1 = li + 1
            idx = (xyz[lj][:,None] * li1 + xyz[li]).transpose(2,0,1)
            dat.append(idx.ravel())
    g_idx = np.hstack(dat).astype(np.int32)
    offsets = np.cumsum([0] + [x.size for x in dat]).astype(np.int32)
    return g_idx, offsets

def init_constant(mol):
    g_idx, offsets = g_pair_idx()
    err = libvhf_rys.RYS_init_constant(
        g_idx.ctypes, offsets.ctypes, mol._env.ctypes,
        ctypes.c_int(mol._env.size), ctypes.c_int(SHM_SIZE))
    if err != 0:
        device_id = cp.cuda.device.get_device_id()
        raise RuntimeError(f'CUDA kernel initialization on device {device_id}')

def _make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond, cutoff, tile):
    n_groups = len(l_ctr_bas_loc) - 1
    ntiles = tile_q_cond.shape[0]
    tile_mappings = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            i0 = ish0 // tile
            i1 = ish1 // tile
            j0 = jsh0 // tile
            j1 = jsh1 // tile
            sub_tile_q = tile_q_cond[i0:i1,j0:j1]
            mask = sub_tile_q > cutoff
            if i == j:
                mask = cp.tril(mask)
            t_ij = (cp.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                    cp.arange(j0, j1, dtype=np.int32))
            idx = cp.argsort(sub_tile_q[mask])[::-1]
            tile_mappings[i,j] = t_ij[mask][idx]
    return tile_mappings

def _make_tril_pair_mappings(l_ctr_bas_loc, q_cond, cutoff, tile=4):
    nbas = q_cond.shape[0]
    q_cond = q_cond.ravel()
    n_groups = len(l_ctr_bas_loc) - 1
    pair_mappings = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            nish = ish1 - ish0
            njsh = jsh1 - jsh0
            ntiles_i = (nish+tile-1) // tile
            ntiles_j = (njsh+tile-1) // tile
            ish = cp.arange(ish0, ish0+ntiles_i*tile, dtype=np.int32).reshape(ntiles_i,tile)
            jsh = cp.arange(jsh0, jsh0+ntiles_j*tile, dtype=np.int32).reshape(ntiles_j,tile)
            ish = ish[:,None,:,None]
            jsh = jsh[None,:,None,:]
            pair_ij = ish * nbas + jsh
            if i == j:
                pair_ij = pair_ij[(ish >= jsh) & (ish < ish1) & (jsh < jsh1)]
            else:
                pair_ij = pair_ij[(ish < ish1) & (jsh < jsh1)]
            pair_ij = pair_ij[q_cond[pair_ij] > cutoff]
            pair_mappings[i,j] = cp.asarray(pair_ij, dtype=np.int32)
    return pair_mappings

def _make_j_engine_pair_locs(mol):
    ls = mol._bas[:,ANG_OF]
    ll = (ls[:,None]+ls).ravel()
    pair_loc = np.append(0, np.cumsum((ll+1)*(ll+2)*(ll+3)//6))
    return np.asarray(pair_loc, dtype=np.int32)

def quartets_scheme(mol, l_ctr_pattern, with_j, with_k, shm_size=SHM_SIZE):
    raise RuntimeError('deprecated')
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    gout_size = nfi * nfj * nfk * nfl
    g_size = (li+1)*(lj+1)*(lk+1)*(ll+1)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = order // 2 + 1
    jk_cache_size = 0
    if with_j: jk_cache_size += nfi*nfj + nfk*nfl
    if with_k: jk_cache_size += nfi*nfk + nfi*nfl + nfj*nfk + nfj*nfl
    root_g_jk_cache_shared = max(nroots*2 + g_size*3, jk_cache_size)
    unit = root_g_jk_cache_shared + ij_prims + 9
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    while gout_stride < 16 and gout_size / (gout_stride*GOUT_WIDTH) > 1:
        n //= 2
        gout_stride *= 2
    return n, gout_stride

def _j_engine_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nmax = max(lij, lkl)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    g_size = (lij+1)*(lkl+1)
    nf3_ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    nroots = order // 2 + 1

    unit = nroots*2 + g_size*3 + 6
    dm_cache_size = nf3_ij + nf3_kl*2 + (lij+1)*(lkl+1)*(nmax+2)
    gout_size = nf3_ij * nf3_kl
    if dm_cache_size < gout_size:
        unit += dm_cache_size
        shm_size -= nf3_ij * 8
        with_gout = False
    else:
        unit += gout_size
        with_gout = True

    if mol.omega < 0:
        unit += nroots*2
    counts = (shm_size-ij_prims*8) // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride, with_gout

def _nearest_power2(n, return_leq=True):
    '''nearest 2**x that is leq or geq than n.

    Kwargs:
        return_leq specifies that the return is less or equal than n.
        Otherwise, the return is greater or equal than n.
    '''
    if isinstance(n, np.ndarray):
        n = n.astype(int, copy=False)
        if return_leq:
            return 2 ** np.log2(n).astype(int)
        else:
            return 2 ** np.ceil(np.log2(n)).astype(int)

    n = int(n)
    assert n > 0
    if return_leq:
        return 1 << (n.bit_length() - 1)
    else:
        return 1 << ((n-1).bit_length())

def _create_q_cond(mol, uniq_l_ctr, l_ctr_offsets, envs, precision=1e-14):
    '''A fast routine to estimate the Schwarz inequality condition sqrt(absmax( (ij|ij) )).
    Note the high angular momentum bases are excluded.
    '''
    from gpu4pyscf.pbc.gto import int1e
    gout_width = 60
    omega = mol.omega
    ls = np.arange(LMAX+1)
    li = ls[:,None]
    lj = ls
    lij = li + lj
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nroots = lij + 1
    if omega < 0:
        nroots *= 2
    unit = (li+1)*(lj+1)*2 + (li+1)*(lj+1)*(lij+1) + 6 + nroots*4
    nsp_max = _nearest_power2(SHM_SIZE // (unit*4))
    gout_size = nfi * nfj
    gout_stride = (gout_size+gout_width-1) // gout_width
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    nsp_per_block = THREADS // gout_stride
    # min(nsp_per_block, nsp_max)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    gout_stride = THREADS // nsp_per_block
    shm_size = nsp_per_block * (unit * 4)
    max_shm_size = shm_size.max()

    ovlp_mask = int1e._shell_overlap_mask(mol, precision=precision**2)
    nbas = np.uint32(mol.nbas)
    assert nbas < 65535
    uniq_l = uniq_l_ctr[:,0]
    bas_ij_idx = [] # The effective shell pair = ish*nbas+jsh
    shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
    sp0 = sp1 = 0
    for i, li in enumerate(uniq_l):
        for j, lj in enumerate(uniq_l[:i+1]):
            if li > LMAX or lj > LMAX:
                continue
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.uint32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.uint32)
            mask = ovlp_mask[ish0:ish1,jsh0:jsh1]
            idx = (ish[:,None] * nbas + jsh)[mask]
            nshl_pair = len(idx)
            bas_ij_idx.append(idx)
            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = THREADS // gout_stride[li, lj] * 8
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))
    ovlp_mask = None
    shl_pair_offsets.append(np.int32(sp1))
    shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
    bas_ij_idx = cp.array(cp.hstack(bas_ij_idx), dtype=np.uint32)

    nbatches_shl_pair = len(shl_pair_offsets) - 1
    q_out = cp.full((nbas, nbas), -700, dtype=np.float32)
    s_out = None
    s_out_ptr = lib.c_null_ptr()
    lr_factor = sr_factor = 1
    if omega < 0:
        # FIXME: To avoid changing the CUDA kernel function signature,
        # temporarily attach the extra information to the s_estimator array and
        # pass it along with s_estimator.
        # This is a workaround and should be addressed in the future.
        s_out = cp.full((nbas+2, nbas), -700, dtype=np.float32)
        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')
        s_out[nbas] = cp.asarray(diffuse_exps, dtype=np.float32)
        s_out[nbas+1] = cp.asarray(diffuse_ctr_coef, dtype=np.float32)
        s_out_ptr = ctypes.cast(s_out.data.ptr, ctypes.c_void_p)
        lr_factor = 0
    if omega > 0:
        sr_factor = 0
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    libvhf_rys.int2e_qcond_estimator(
        ctypes.cast(q_out.data.ptr, ctypes.c_void_p),
        s_out_ptr,
        ctypes.byref(envs),
        ctypes.c_int(max_shm_size),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.c_double(omega),
        ctypes.c_double(lr_factor),
        ctypes.c_double(sr_factor))
    return q_out, s_out
