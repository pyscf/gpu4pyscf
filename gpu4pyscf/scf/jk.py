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
    load_library, condense, transpose_sum, hermi_triu, asarray, ndarray)
from gpu4pyscf.__config__ import num_devices, shm_size
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import (
    extract_pgto_params, _scale_sp_ctr_coeff, RysIntEnvVars, SortedGTO)

__all__ = [
    'get_jk', 'get_j', 'get_k',
]

libvhf_rys = load_library('libgvhf_rys')
libvhf_rys.RYS_build_jk.restype = ctypes.c_int
libvhf_rys.RYS_build_k.restype = ctypes.c_int
libvhf_rys.cuda_version.restype = ctypes.c_int
CUDA_VERSION = libvhf_rys.cuda_version()
libgint = load_library('libgint')

libvhf_rys.RYS_init_constant()

PTR_BAS_COORD = 7
LMAX = 4
TILE = 6
QUEUE_DEPTH = 262144
SHM_SIZE = shm_size - 1024
del shm_size
GOUT_WIDTH = 42
THREADS = 256
GROUP_SIZE = 256
Q_COND_MARGIN = 4.

libvhf_rys.RYS_build_k_init.restype = ctypes.c_int
libvhf_rys.RYS_build_jk_init.restype = ctypes.c_int

def get_jk(mol, dm, hermi=0, vhfopt=None, with_j=True, with_k=True, verbose=None):
    '''Compute J, K matrices
    '''
    assert with_j or with_k
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    dm = cp.asarray(dm, order='C')
    nao_orig = dm.shape[-1]
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

def get_k(mol, dm, hermi=0, vhfopt=None, omega=None, lr_factor=None, sr_factor=None):
    '''Compute K matrix
    '''
    log = logger.new_logger(mol)
    cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    dm = cp.asarray(dm, order='C')
    nao_orig = dm.shape[-1]
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    dms = cp.asarray(dms, order='C')

    vk = vhfopt.get_k(dms, hermi, log, omega, lr_factor, sr_factor)
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

    dm = cp.asarray(dm, order='C')
    nao_orig = dm.shape[-1]
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
    Fused kernel to perform 'pi,nij,qj->npq'
    '''
    spherical_matrix = cp.asarray(spherical_matrix, order='C')
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
    Fused kernel to perform 'pi,npq,qj->nij'
    '''
    cartesian_matrix = cp.asarray(cartesian_matrix, order='C')
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

def apply_coeff_C_mat(right_matrix, mol, sorted_mol, uniq_l_ctr,
                      l_ctr_offsets, ao_idx, l_ctr_paddings=None):
    '''
    Sort AO and perform sph2cart transformation (if needed) for the second last axis
    Fused kernel to perform 'pi,nij->npj'
    '''
    right_matrix = cp.asarray(right_matrix, order='C')
    ndim = right_matrix.ndim
    if ndim == 2:
        right_matrix = right_matrix[None]
    nao, n_second = right_matrix.shape[1:]
    assert nao == mol.nao
    n_cartesian = sorted_mol.nao

    output_complex = False
    if right_matrix.dtype == np.complex128:
        right_matrix = right_matrix.view(np.float64)
        right_matrix = right_matrix.reshape(-1,nao,n_second,2)
        right_matrix = right_matrix.transpose(3,0,1,2).reshape(-1,nao,n_second)
        output_complex = True
    else:
        assert right_matrix.dtype == np.float64
    counts = len(right_matrix)

    l_ctr_count = np.asarray(l_ctr_offsets[1:] - l_ctr_offsets[:-1], dtype = np.int32)
    l_ctr_l = np.asarray(uniq_l_ctr[:,0].copy(), dtype = np.int32)
    if l_ctr_paddings is None:
        l_ctr_pad_counts = np.zeros_like(l_ctr_count)
    else:
        l_ctr_pad_counts = np.asarray(l_ctr_paddings, dtype=np.int32)
    ao_idx = cp.asarray(ao_idx, dtype=np.int32)
    stream = cp.cuda.get_current_stream()

    out = cp.zeros((counts, n_cartesian, n_second), order = "C")
    for i in range(counts):
        libgint.cart2sph_C_mat_with_padding(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(out[i].data.ptr, ctypes.c_void_p),
            ctypes.cast(right_matrix[i].data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_second),
            ctypes.c_int(l_ctr_l.shape[0]),
            l_ctr_l.ctypes.data_as(ctypes.c_void_p),
            l_ctr_count.ctypes.data_as(ctypes.c_void_p),
            l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(ao_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_bool(mol.cart),
        )

    if output_complex:
        outR, outI = out.reshape(2, -1, n_cartesian, n_second)
        out = outR.astype(np.complex128)
        out.imag = outI

    if ndim == 2:
        out = out[0]
    return out

class _VHFOpt:
    def __init__(self, mol, direct_scf_tol=1e-13, tile=TILE):
        self.mol = mol
        self.direct_scf_tol = direct_scf_tol
        self.h_shls = None
        self.tile = tile

        # Hold cache on GPU devices
        self.sorted_mol = None
        self._rys_envs = {}

    def reset(self, mol):
        self.mol = mol
        self.sorted_mol = None
        self._rys_envs = {}

    def build(self, group_size=None, verbose=None):
        log = logger.new_logger(self.mol, verbose)
        cput0 = log.init_timer()
        mol = self.sorted_mol = SortedGTO.from_mol(
            self.mol, decontract=True, diffuse_cutoff=0.3)
        l_ctr_counts = mol.l_ctr_counts

        # very high angular momentum basis are processed on CPU
        lmax = mol.uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[mol.uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        self.bas_pair_cache = _cache_q_cond_and_non0pairs(
            mol, self.rys_envs, self.direct_scf_tol, self.tile)
        log.timer('Initialize q_cond', *cput0)
        return self

    def apply_coeff_C_mat_CT(self, spherical_matrix):
        '''
        Unsort AO and perform sph2cart transformation (if needed) for the last 2 axes
        Fused kernel to perform 'ip,npq,qj->nij'
        '''
        if self.sorted_mol is None:
            self.build()
        return self.sorted_mol.apply_C_mat_CT(spherical_matrix)

    def apply_coeff_CT_mat_C(self, cartesian_matrix):
        '''
        Sort AO and perform cart2sph transformation (if needed) for the last 2 axes
        Fused kernel to perform 'ip,npq,qj->nij'
        '''
        if self.sorted_mol is None:
            self.build()
        return self.sorted_mol.apply_CT_mat_C(cartesian_matrix)

    def apply_coeff_C_mat(self, right_matrix):
        '''
        Sort AO and perform sph2cart transformation (if needed) for the second last axis
        Fused kernel to perform 'ip,npq->niq'
        '''
        if self.sorted_mol is None:
            self.build()
        return self.sorted_mol.apply_C_dot(right_matrix)

    @property
    def q_cond(self):
        raise RuntimeError('deprecated')

    @property
    def s_estimator(self):
        raise RuntimeError('deprecated')

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
        return self.sorted_mol.ctr_coeff

    def get_jk(self, dms, hermi, verbose):
        '''
        Build JK for the sorted_mol. Density matrices dms and the return JK
        matrices are all corresponding to the sorted_mol
        '''
        log = logger.new_logger(self.mol, verbose)
        if self.sorted_mol is None:
            self.build()
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        ao_loc = mol.ao_loc
        uniq_l = mol.uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(mol.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        assert dms.ndim == 3 and dms.shape[-1] == ao_loc[-1]
        dm_cond = condense('absmax', dms, ao_loc)
        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        dm_penalty = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')

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
            err = libvhf_rys.RYS_build_jk_init(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError('RYS build_jk CUDA kernel initialization failed')
            dms = cp.asarray(dms) # transfer to current device
            dm_cond = cp.asarray(dm_cond)

            if hermi == 0:
                # Contract the tril and triu parts separately
                dms = cp.vstack([dms, dms.transpose(0,2,1)])
            n_dm, nao = dms.shape[:2]
            vj = cp.zeros(dms.shape)
            vk = cp.zeros(dms.shape)
            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            rys_envs = self.rys_envs

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            # An additional integer to count for the proccessed pair_ijs
            pool = cp.empty(workers*QUEUE_DEPTH+3, dtype=np.int32)

            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            kern = libvhf_rys.RYS_build_jk

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(vj.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    mol._bas.ctypes, mol._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()

            if hermi == 1:
                vj *= 2.
                vk = transpose_sum(vk)
            elif hermi == 2:
                vj[:] = 0
                #:vk = vk - vk.transpose(0,2,1)
                vk = transpose_sum(vk, hermi=2)
            else:
                vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
                vj += vjT.transpose(0,2,1)
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
            return vj, vk, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        if self.h_shls:
            dms = dms.get()
            dm_cond = None
        else:
            dms = dm_cond = None
        vk = multi_gpu.array_reduce([x[1] for x in results], inplace=True)
        vj = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        vj = transpose_sum(vj)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[2] for x in results))
            _TimingCollector.summary(log.debug1, (x[3] for x in results))

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
        log = logger.new_logger(self.mol, verbose)
        if self.sorted_mol is None:
            self.build()
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        ao_loc = mol.ao_loc
        n_dm, nao = dms.shape[:2]
        assert dms.ndim == 3 and nao == ao_loc[-1]
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        dm_penalty = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        uniq_l_ctr = mol.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(mol.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        if isinstance(dms, cp.ndarray):
            dms = dms.get()
        pair_loc = _make_j_engine_pair_locs(mol)
        dm_xyz = np.empty(pair_loc[-1])
        libvhf_rys.transform_cart_to_xyz(
            dm_xyz.ctypes, dms.ctypes, ao_loc.ctypes, pair_loc.ctypes,
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')

        tasks = [(i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1)]
        schemes = {t: _j_engine_quartets_scheme(mol, uniq_l_ctr[list(t)]) for t in tasks}
        tasks = iter(tasks)

        libvhf_rys.RYS_init_rysj_constant.restype = ctypes.c_int
        libvhf_rys.RYS_build_j.restype = ctypes.c_int

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
            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}

            err = libvhf_rys.RYS_init_rysj_constant(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError('CUDA kernel initialization')

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)

            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            kern = libvhf_rys.RYS_build_j

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = schemes[task]
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    mol._bas.ctypes, mol._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_build_j kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dm_xyz, dm_cond), non_blocking=True)
        dm_xyz = dm_cond = None

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            _TimingCollector.summary(log.debug1, (x[2] for x in results))

        vj_xyz = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
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

    def get_k(self, dms, hermi, verbose, omega, lr_factor, sr_factor):
        '''
        Build K matrix for the sorted_mol. Density matrices dms and the return K
        matrix are all corresponding to the sorted_mol
        '''
        log = logger.new_logger(self.mol, verbose)
        if self.sorted_mol is None:
            self.build()
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        ao_loc = mol.ao_loc
        uniq_l = mol.uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(mol.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        assert dms.ndim == 3 and dms.shape[-1] == ao_loc[-1]
        dm_cond = condense('absmax', dms, ao_loc)
        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        dm_penalty = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')

        omega, lr_factor, sr_factor = _check_rsh_factors(self.mol, omega, lr_factor, sr_factor)

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
            err = libvhf_rys.RYS_build_k_init(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError('RYS build_k CUDA kernel initialization failed')
            dms = cp.asarray(dms) # transfer to current device
            dm_cond = cp.asarray(dm_cond)

            if hermi == 0:
                # Contract the tril and triu parts separately
                dms = cp.vstack([dms, dms.transpose(0,2,1)])
            n_dm, nao = dms.shape[:2]
            vk = cp.zeros(dms.shape)
            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            rys_envs = self.rys_envs

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+3, dtype=np.int32)

            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            kern = libvhf_rys.RYS_build_k

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.c_double(omega),
                    ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
                    ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    mol._bas.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()

            if hermi == 1:
                vk = transpose_sum(vk)
            elif hermi == 2:
                #:vk = vk - vk.transpose(0,2,1)
                vk = transpose_sum(vk, hermi=2)
            else:
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
            return vk, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        if self.h_shls:
            dms = dms.get()
            dm_cond = None
        else:
            dms = dm_cond = None
        vk = multi_gpu.array_reduce([x[0] for x in results], inplace=True)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            _TimingCollector.summary(log.debug1, (x[2] for x in results))

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
    unit = root_g_jk_cache_shared + ij_prims + 8
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

def _cache_q_cond_and_non0pairs(mol, rys_envs, precision=1e-14, tile=4, tril=True):
    '''A fast routine to estimate the Schwarz inequality condition
    log(sqrt(absmax( (ij|ij) ))).
    Note the high angular momentum bases are excluded.
    '''
    from gpu4pyscf.pbc.gto import int1e
    from gpu4pyscf.pbc.scf.rsjk import libpbc, _group_by_split_points
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

    SIZEOF_FLOAT = ctypes.sizeof(ctypes.c_float)
    gout_width = 29
    unit = (li+1)*(lj+1)*2 + (li+1)*(lj+1)*(lij+1) + 6 + nroots*2
    shm_size = 1024 * 48 - 1024
    nsp_max = _nearest_power2(shm_size // (unit*SIZEOF_FLOAT))
    gout_size = nfi * nfj
    gout_stride = (gout_size+gout_width-1) // gout_width
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    nsp_per_block = THREADS // gout_stride
    # min(nsp_per_block, nsp_max)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    gout_stride = THREADS // nsp_per_block
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    shm_size = nsp_per_block * (unit*SIZEOF_FLOAT)
    # (pp|pp) requires more shm than this estimation. 5888 is the required size
    max_shm_size = max(shm_size.max(), 5888*SIZEOF_FLOAT)

    pair_ij_kern = libpbc.PBCsort_pair_ij
    pair_ij_kern.restype = ctypes.c_int

    l_ctr_offsets = np.append(0, np.cumsum(mol.l_ctr_counts))
    n = mol.l_ctr_counts.max()
    pair_buf = cp.empty(n**2, dtype=np.int64)
    ovlp_mask = int1e._shell_overlap_mask(mol, precision=precision**2)
    if tril:
        ovlp_mask = cp.tril(ovlp_mask)
    ovlp_mask = ovlp_mask.ravel()
    nbas = mol.nbas
    uniq_l = mol.uniq_l_ctr[:,0]
    n_groups = np.count_nonzero(uniq_l <= LMAX)
    if tril:
        pair_keys = ((i, j) for i in range(n_groups) for j in range(i+1))
    else:
        pair_keys = ((i, j) for i in range(n_groups) for j in range(n_groups))
    bas_ij_cache = {} # The effective shell pair = ish*nbas+jsh
    shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
    sp0 = sp1 = 0
    for i, j in pair_keys:
        li = uniq_l[i]
        lj = uniq_l[j]
        ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
        jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
        ish = cp.arange(ish0, ish1, dtype=np.uint32)
        jsh = cp.arange(jsh0, jsh1, dtype=np.uint32)
        nish = len(ish)
        njsh = len(jsh)
        pair_ij = ndarray(nish*njsh, dtype=np.int64, buffer=pair_buf)
        err = pair_ij_kern(
            ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(ish.data.ptr, ctypes.c_void_p),
            ctypes.cast(jsh.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nish), ctypes.c_int(njsh),
            ctypes.c_int(nbas), ctypes.c_int(tile))
        pair_ij = pair_ij[ovlp_mask[pair_ij]]
        bas_ij_cache[i,j] = cp.asarray(pair_ij, dtype=np.uint32)
        nshl_pair = len(pair_ij)
        sp0, sp1 = sp1, sp1 + nshl_pair
        nsp_per_block = THREADS // gout_stride[li, lj] * 8
        shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))
    ovlp_mask = None
    shl_pair_offsets.append(np.int32(sp1))
    shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
    bas_ij_counts = [len(x) for x in bas_ij_cache.values()]
    bas_ij_cum = np.cumsum(bas_ij_counts)
    bas_ij_idx = cp.array(cp.hstack(bas_ij_cache.values()), dtype=np.uint32)

    lr_factor = sr_factor = 1
    if omega < 0:
        lr_factor = 0
    if omega > 0:
        sr_factor = 0
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    q_out = cp.empty(len(bas_ij_idx), dtype=np.float32)
    libvhf_rys.int2e_qcond_estimator.restype = ctypes.c_int
    err = libvhf_rys.int2e_qcond_estimator(
        ctypes.cast(q_out.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs),
        ctypes.c_int(max_shm_size),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.c_double(omega),
        ctypes.c_double(lr_factor),
        ctypes.c_double(sr_factor))
    if err != 0:
        raise RuntimeError('int2e_qcond_estimator kernel failed')

    if omega < 0:
        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')
        diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
        diffuse_ctr_coef = cp.asarray(diffuse_ctr_coef, dtype=np.float32)
        s_out = cp.empty(len(bas_ij_idx), dtype=np.float32)
        libvhf_rys.fill_s_estimator.restype = ctypes.c_int
        err = libvhf_rys.fill_s_estimator(
            ctypes.cast(s_out.data.ptr, ctypes.c_void_p),
            ctypes.byref(rys_envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_ctr_coef.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_double(omega))
        if err != 0:
            raise RuntimeError('fill_s_estimator kernel failed')

    split_points = cp.arange(math.log(precision), 2., Q_COND_MARGIN)
    q_cond_cache = {}
    q_cond = cp.split(q_out, bas_ij_cum[:-1])
    if omega < 0:
        s_cond = cp.split(s_out, bas_ij_cum[:-1])
    for i, key in enumerate(bas_ij_cache):
        idx = _group_by_split_points(q_cond[i], split_points)
        pair_ij = bas_ij_cache[key][idx]
        q_cond_ij = q_cond[i][idx]
        s_cond_ij = q_cond_ij
        if omega < 0:
            s_cond_ij = s_cond[i][idx]
        q_cond_cache[key] = pair_ij, q_cond_ij, s_cond_ij
    return q_cond_cache

def _check_rsh_factors(mol, omega, lr_factor, sr_factor):
    '''
    The exchange operator of the range-separation hybrid functional is

        lr_factor * erf(|omega|r12)/r12 + sr_factor * erfc(|omega|r12)/r12.

    This function returns (omega, lr_factor, sr_factor) that are compatible for
    rys_contract_k CUDA kernel. In this kernel, omega<0 indicates that SR
    contribution needs to be evaluated, which will allocate 2*nrys roots.
    '''
    if omega is None:
        omega = mol.omega

    if lr_factor is None and sr_factor is None:
        # This is the convention employed by libcint, which uses the sign of
        # omega to determine the lr_factor and sr_factor
        if omega == 0:
            lr_factor = sr_factor = 1
        elif omega < 0: # short-range Coulomb
            lr_factor, sr_factor = 0, 1
        else: # long-range
            lr_factor, sr_factor = 1, 0
    elif lr_factor is None: # short-range
        if omega == 0:
            lr_factor = sr_factor
        else:
            # omega<0 is allowed, following libcint convention
            lr_factor = 0
    elif sr_factor is None: # long-range
        if omega == 0:
            sr_factor = lr_factor
        else:
            # libcint convention requires omega>0 for long-range
            assert omega > 0
            sr_factor = 0
    else:
        # When lr_factor and sr_factor are both provided, omega >= 0 is enforced
        if omega == 0:
            assert lr_factor == sr_factor
        elif lr_factor == sr_factor: # identical to full-range
            omega = 0

    if sr_factor != 0 and omega != 0:
        # rys_contract_k kernel follows libcint convention, which uses omega<0
        # to indicate SR Coulomb potential
        omega = -abs(omega)
    return omega, lr_factor, sr_factor

class _TimingCollector:
    def __init__(self, timer):
        self.timer = timer
        self.collection = {}

    def collect(self, key, t1, msg):
        t1, t1p = self.timer(msg, *t1), t1
        if key not in self.collection:
            self.collection[key] = 0
        self.collection[key] += cp.cuda.get_elapsed_time(t1p[2], t1[2]) * 1e3
        return t1

    @staticmethod
    def summary(logger, timing_collectors):
        timing_counter = Counter()
        for t in timing_collectors:
            timing_counter += t.collection
        for key, t in timing_counter.items():
            logger(f'{key} wall time {t:.2f}')
