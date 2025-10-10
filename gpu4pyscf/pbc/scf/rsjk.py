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
Compute K Matrices with Periodic Boundary Conditions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from collections import Counter
from pyscf import lib, gto
from pyscf.scf import _vhf
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.__config__ import _streams, num_devices, shm_size
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    condense, transpose_sum, dist_matrix, contract, asarray)
from gpu4pyscf.gto.mole import group_basis, groupby, extract_pgto_params
from gpu4pyscf.scf.jk import (
    libvhf_rys, RysIntEnvVars, _scale_sp_ctr_coeff,
    _nearest_power2, apply_coeff_C_mat_CT, apply_coeff_CT_mat_C,
    PTR_BAS_COORD, LMAX, QUEUE_DEPTH, SHM_SIZE, GOUT_WIDTH)
from gpu4pyscf.pbc.df.ft_ao import libpbc, most_diffuse_pgto
from gpu4pyscf.pbc.dft.multigrid_v2 import _unique_image_pair

__all__ = [
    'get_k',
]

libpbc.PBC_build_k.restype = ctypes.c_int
libpbc.PBC_build_k_init(ctypes.c_int(SHM_SIZE))

OMEGA = 0.3

def get_k(cell, dm, hermi=0, kpts=None, omega=None, vhfopt=None, verbose=None):
    '''Compute K matrix
    '''
    if vhfopt is None:
        vhfopt = PBCJKmatrixOpt(cell, omega).build()
    vk = vhfopt.get_k(dm, hermi, kpts, verbose)
    return vk

class PBCJKmatrixOpt:
    def __init__(self, cell, omega=None):
        self.cell = cell
        if omega is None: # TODO: dynamically determine omega based on rcut?
            omega = OMEGA
        self.omega = omega
        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.h_shls = None
        self.supmol = None
        # Hold cache on GPU devices
        self._rys_envs = {}
        self._q_cond = {}
        self._s_estimator = {}

    def build(self, group_size=None, verbose=None):
        cell = self.cell
        log = logger.new_logger(cell, verbose)
        cput0 = log.init_timer()
        cell, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = group_basis(
            cell, 1, group_size, sparse_coeff=True)
        self.sorted_cell = cell
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = np.asarray(l_ctr_pad_counts, dtype=np.int32)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        # FIXME: should the supmol be regrouped based on l?
        supmol = self.supmol = ExtendedMole.from_cell(cell, self.omega)

        lmax = uniq_l_ctr[:,0].max()
        if lmax > LMAX:
            raise NotImplementedError('basis set with h functions')

        # TODO: approx with overlap mask
        nbas = supmol.nbas
        ao_loc = supmol.ao_loc
        q_cond = np.empty((nbas,nbas))
        intor = supmol._add_suffix('int2e')
        _vhf.libcvhf.CVHFnr_int2e_q_cond(
            getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
            q_cond.ctypes, ao_loc.ctypes,
            supmol._atm.ctypes, ctypes.c_int(supmol.natm),
            supmol._bas.ctypes, ctypes.c_int(supmol.nbas), supmol._env.ctypes)
        q_cond = np.log(q_cond + 1e-300).astype(np.float32)
        self.q_cond_cpu = q_cond

        # CVHFnr_sr_int2e_q_cond in pyscf is inaccurate for upper bound estimation.
        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(supmol, 'diffuse')
        s_estimator = np.empty((nbas+2,nbas), dtype=np.float32)
        # FIXME: To avoid changing the CUDA kernel function signature,
        # temporarily attach the extra information to the s_estimator array and
        # pass it along with s_estimator.
        # This is a workaround and should be addressed in the future.
        diffuse_exps = diffuse_exps.astype(np.float32)
        diffuse_ctr_coef = diffuse_ctr_coef.astype(np.float32)
        s_estimator[nbas] = diffuse_exps
        s_estimator[nbas+1] = diffuse_ctr_coef
        libvhf_rys.sr_eri_s_estimator_v2(
            s_estimator.ctypes, ctypes.c_float(supmol.omega),
            diffuse_exps.ctypes, diffuse_ctr_coef.ctypes,
            supmol._atm.ctypes, ctypes.c_int(supmol.natm),
            supmol._bas.ctypes, ctypes.c_int(supmol.nbas), supmol._env.ctypes)
        self.s_estimator_cpu = s_estimator
        log.timer('Initialize q_cond', *cput0)
        return self

    def reset(self, cell):
        self.cell = cell
        self.supmol = None
        self._q_cond = {}
        self._s_estimator = {}

    @multi_gpu.property(cache='_q_cond')
    def q_cond(self):
        return asarray(self.q_cond_cpu)

    @multi_gpu.property(cache='_s_estimator')
    def s_estimator(self):
        return asarray(self.s_estimator_cpu)

    @multi_gpu.property(cache='_rys_envs')
    def rys_envs(self):
        supmol = self.supmol
        atm = asarray(supmol._atm)
        bas = asarray(supmol._bas)
        env = asarray(_scale_sp_ctr_coeff(supmol))
        ao_loc = asarray(supmol.ao_loc)
        return RysIntEnvVars.new(supmol.natm, supmol.nbas, atm, bas, env, ao_loc)

    def estimate_cutoff_with_penalty(self):
        cell = self.cell
        vol = cell.vol
        rcut = cell.rcut
        omega = self.omega
        exp_min, _, l = most_diffuse_pgto(cell)
        theta = 1./(1./exp_min + omega**-2)
        lsum = l * 4 + 1
        lat_unit = vol**(1./3)
        rad = rcut / lat_unit + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = 2*np.pi*(rcut+lat_unit)*lsum/(vol*theta) + surface
        cutoff = cell.precision / lattice_sum_factor
        logger.debug1(cell, 'int3c_kernel integral omega=%g theta=%g cutoff=%g',
                      omega, theta, cutoff)
        # When exp_min is small, the lattice sum over j and k in (ij|kl) would
        # contribute to the kl-pair near the cutoff edges. Accurate estimation
        # for their contributions is hard to derive. Numerical tests show that
        # the contribution is approximately proportional to 1/(exp_min**3*vol**2).
        double_lat_sum_penalty = 1
        if theta * lat_unit < 2:
            double_lat_sum_penalty = max(1, (25.13/(exp_min*lat_unit**2))**3)
        return cutoff / double_lat_sum_penalty

    def get_k(self, dm, hermi, kpts, kpts_band=None, verbose=None):
        '''
        Build K for the sorted_mol over the sampled k-points.
        Return a (*, nkpts, nao, nao) array.

        If the "kpts" is supplied as None or [[0,0,0]] (the gamma point), the K
        matrix is still evaluated as the k-point sampling case. The "nkpts"
        dimension is set to 1
        '''
        assert hermi == 1
        cell = self.cell
        sorted_cell = self.sorted_cell
        nao_orig = cell.nao
        nao = sorted_cell.nao
        supmol = self.supmol

        dm = asarray(dm, order='C')
        dms = dm.reshape(-1,nao_orig,nao_orig)
        #:dms = cp.einsum('pi,nij,qj->npq', self.coeff, dms, self.coeff)
        dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, self.uniq_l_ctr,
                                   self.l_ctr_offsets, self.ao_idx)

        is_gamma_point = kpts is None or is_zero(kpts)
        if is_gamma_point:
            assert dms.dtype == np.float64
            nimgs = len(supmol.Ls)
            nkpts = 1
            dms = dms.reshape(-1, nao, nao)
            n_dm = len(dms)
            ish_cell0 = supmol.ao_mapping % nao
            dms = dms[:,:,ish_cell0]
        else:
            expLk = cp.exp(1j * asarray(supmol.Ls).dot(asarray(kpts).T))
            nimgs, nkpts = expLk.shape
            dms = dms.reshape(-1, nkpts, nao, nao)
            dms = contract('skpq,Lk->spLq', dms, expLk)
            expLk = None
            n_dm = len(dms)
            if 1:
                #FIXME: the imaginary part?
                dms = dms.real
            dms = dms.reshape(n_dm, nao, nimgs*nao)[:,:,supmol.ao_mapping]
        dms = cp.asarray(dms, order='C')

        ao_loc_cpu = supmol.ao_loc
        ao_loc = asarray(ao_loc_cpu)
        nao_supmol = ao_loc_cpu[-1]
        uniq_l_ctr = self.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = self.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        dm_cond = condense('absmax', dms, ao_loc[:sorted_cell.nbas+1], ao_loc)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max().get())
        log_cutoff = math.log(self.estimate_cutoff_with_penalty())

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            log = logger.new_logger(cell, verbose)
            t0 = log.init_timer()
            dms = cp.asarray(dms)
            dm_cond = cp.asarray(dm_cond)

            q_cond = cp.asarray(self.q_cond)
            s_estimator = cp.asarray(self.s_estimator)
            pair_ij_mappings, pair_kl_mappings = _make_tril_pair_mappings(
                supmol, l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
            bas_mask_idx = cp.asarray(supmol.bas_mask_idx)
            nimgs = len(supmol.Ls)
            nimgs_uniq_pair = len(supmol.double_latsum_Ts)
            if is_gamma_point:
                Ts_ji_lookup = cp.zeros_like(supmol.Ts_ji_lookup)
                vk = cp.zeros((n_dm, nao, nao))
            else:
                Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup)
                vk = cp.zeros((n_dm, nimgs_uniq_pair, nao, nao))

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)

            timing_counter = Counter()
            kern_counts = 0
            kern = libpbc.PBC_build_k
            rys_envs = self.rys_envs

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping = pair_ij_mappings[i,j]
                pair_kl_mapping = pair_kl_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                err = kern(
                    ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao_supmol),
                    rys_envs, (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_estimator.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(sorted_cell.nbas),
                    supmol._atm.ctypes, ctypes.c_int(supmol.natm),
                    supmol._bas.ctypes, ctypes.c_int(supmol.nbas),
                    supmol._env.ctypes)
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

            cp.cuda.get_current_stream().synchronize()
            if not is_gamma_point:
                scaled_kpts = cell.lattice_vectors().dot(kpts.T)
                Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
                expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
                expLkz = expLk.view(np.float64).reshape(nimgs, nkpts, 2)
                vk = contract('sLmn,Lkz->skmnz', vk, expLkz)
                vk = vk.view(np.complex128)[:,:,:,:,0]
                vk = vk.reshape(n_dm, nkpts, nao, nao)
            return vk, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)

        kern_counts = 0
        timing_collection = Counter()
        vk_dist = []
        for vk, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            vk_dist.append(vk)

        log = logger.new_logger(cell, verbose)
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        vk = multi_gpu.array_reduce(vk_dist, inplace=True)
        vk = vk.reshape(-1,nao,nao)
        vk = apply_coeff_C_mat_CT(vk, cell, sorted_cell, self.uniq_l_ctr,
                                  self.l_ctr_offsets, self.ao_idx)
        vk = vk.reshape(dm.shape)
        return vk

    def get_ek_ip1(self, dm, hermi, kpts, kpts_band=None, verbose=None):
        cell = self.cell
        sorted_cell = self.sorted_cell
        nao_orig = cell.nao
        nao = sorted_cell.nao
        supmol = self.supmol

        dm = asarray(dm, order='C')
        dms = dm.reshape(-1,nao_orig,nao_orig)
        #:dms = cp.einsum('pi,nij,qj->npq', self.coeff, dms, self.coeff)
        dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, self.uniq_l_ctr,
                                   self.l_ctr_offsets, self.ao_idx)

        double_latsum_Ts = supmol.double_latsum_Ts
        is_gamma_point = kpts is None or is_zero(kpts)
        if is_gamma_point:
            expLk = cp.ones((1, 1))
            n_Ts, nkpts = expLk.shape
        else:
            scaled_kpts = cell.lattice_vectors().dot(kpts.T)
            Ts = cp.asarray(double_latsum_Ts, dtype=np.float64)
            expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
            n_Ts, nkpts = expLk.shape
        dms = dms.reshape(-1, nkpts, nao, nao)
        n_dm = len(dms)
        dms = contract('skpq,Lk->spLq', dms, expLk)
        expLk = None
        dms = dms.real
        dms = cp.asarray(dms, order='C')

        ao_loc_cpu = supmol.ao_loc
        ao_loc = asarray(ao_loc_cpu)
        nao_supmol = ao_loc_cpu[-1]
        uniq_l_ctr = self.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = self.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        assert hermi == 1
        if is_gamma_point:
            dm_cond = condense('absmax', dms.reshape(n_dm, nao, nao),
                               ao_loc[:sorted_cell.nbas+1])
            ish_cell0 = supmol.bas_mask_idx % sorted_cell.nbas
            dm_cond = dm_cond[ish_cell0[:,None], ish_cell0]
        else:
            dm_cond = _compressed_dm_cond(supmol, dms)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max().get())
        log_cutoff = math.log(self.estimate_cutoff_with_penalty())

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            log = logger.new_logger(cell, verbose)
            t0 = log.init_timer()
            dms = cp.asarray(dms)
            dm_cond = cp.asarray(dm_cond)

            q_cond = cp.asarray(self.q_cond)
            s_estimator = cp.asarray(self.s_estimator)
            pair_ij_mappings, pair_kl_mappings = _make_tril_pair_mappings(
                supmol, l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
            bas_mask_idx = cp.asarray(supmol.bas_mask_idx)
            nimgs = len(supmol.Ls)
            nimgs_uniq_pair = len(supmol.double_latsum_Ts)
            if is_gamma_point:
                Ts_ji_lookup = cp.zeros_like(supmol.Ts_ji_lookup)
            else:
                Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup)
            ek = cp.zeros((cell.natm, 3))

            t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)

            timing_counter = Counter()
            kern_counts = 0
            kern = libpbc.PBC_build_k_ip1
            rys_envs = self.rys_envs

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping = pair_ij_mappings[i,j]
                pair_kl_mapping = pair_kl_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                err = kern(
                    ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao_supmol),
                    rys_envs, (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_estimator.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(sorted_cell.nbas),
                    supmol._atm.ctypes, ctypes.c_int(supmol.natm),
                    supmol._bas.ctypes, ctypes.c_int(supmol.nbas),
                    supmol._env.ctypes)
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
            return ek, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)

        kern_counts = 0
        timing_collection = Counter()
        ek_dist = []
        for ek, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            ek_dist.append(ek)

        log = logger.new_logger(cell, verbose)
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)
        return ek


class ExtendedMole(gto.Mole):
    '''A super-Mole cluster to mimic periodicity within the unit cell'''
    def __init__(self):
        self.cell = None
        self.Ls = None
        self.precision = None
        # A raw-supmol is a large Mole cluster that consists of repeated unit cells.
        # Many of the shells within the raw-supmol have negligible contributions to the
        # periodicity, and can be eliminated.
        # bas_mask_idx is used to filter out the unnecessary shells from the raw-supmol.
        # supmol._bas == raw_supmol._bas[bas_mask_idx]
        # ao_mapping maps the AOs of the raw-supmol to the trimmed supmol.
        # supmol.ao_labels() == raw_supmol.ao_labels()[ao_mapping]
        self.bas_mask_idx = None
        self.ao_mapping = None
        # double_latsum_Ts stores the unique image pairs for double lattice-sums
        # associated with orbital products.
        # Ts_lookup stores the mapping between the image-pair to the unique
        # image (-img_i + img_j) ~ Ts_lookup[img_j, img_i] == index of Ts
        self.double_latsum_Ts = None
        self.Ts_ji_lookup = None

    @classmethod
    def from_cell(cls, cell, omega, verbose=None):
        log = logger.new_logger(cell, verbose)
        if cell.dimension == 0:
            raise NotImplementedError

        rcut = estimate_rcut(cell, omega)
        Ls = cell.get_lattice_Ls(rcut=rcut.max())
        Ls = Ls[np.linalg.norm(Ls-.1, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug1('Generate supmol with rcut = %g nimgs = %d', rcut, nimgs)

        supmol = cls()
        supmol.__dict__.update(cell.__dict__)
        supmol = pbctools._build_supcell_(supmol, cell, Ls)
        supmol.cell = cell
        supmol.Ls = Ls
        supmol.precision = cell.precision
        supmol._env[gto.PTR_EXPCUTOFF] = -np.log(cell.precision*1e-4)
        supmol.omega = -abs(omega) # Use supmol to handle SR integrals only

        rcut_for_atoms = asarray(groupby(cell._bas[:,gto.ATOM_OF], rcut, 'max'))
        # Search the shortest distance to the reference cell for each atom in the supercell.
        atom_coords = supmol.atom_coords()
        d = dist_matrix(atom_coords, cell.atom_coords())
        mask = cp.any(d < rcut_for_atoms, axis=1).get()
        bas_mask = mask[supmol._bas[:,gto.ATOM_OF]]
        bas_mask[:cell.nbas] = True # Ensure shells in the first image are all included
        bas_mask_idx = np.where(bas_mask)[0]

        ao_loc = supmol.ao_loc
        nao = ao_loc[-1]
        ao_idx_frags = np.split(np.arange(nao), ao_loc[1:-1])
        ao_mapping = np.hstack([ao_idx_frags[i] for i in bas_mask_idx])
        supmol.bas_mask_idx = np.asarray(bas_mask_idx, dtype=np.int32)
        supmol.ao_mapping = np.asarray(ao_mapping, dtype=np.int32)
        supmol._bas = supmol._bas[bas_mask_idx]
        supmol._bas[:,PTR_BAS_COORD] = supmol._atm[supmol._bas[:,gto.ATOM_OF],gto.PTR_COORD]
        logger.debug1(supmol, 'trim supmol %d shells -> %d shells, %d AOs -> %d AOs',
                      nimgs*cell.nbas, supmol.nbas, nao, len(ao_mapping))

        translation_vectors = asarray(np.linalg.solve(cell.lattice_vectors().T, Ls.T).T)
        translation_vectors = cp.asarray(translation_vectors.round(), dtype=np.int32)
        supmol.double_latsum_Ts, inverse = _unique_image_pair(translation_vectors)
        supmol.Ts_ji_lookup = cp.asarray(inverse, order='C', dtype=np.int32)
        return supmol

def estimate_rcut(cell, omega, precision=None):
    '''Estimate rcut for 2e SR-integrals

    This function is generally based on the implementation of
    pyscf.pbc.scf.rsjk.estimate_rcut with small modifications in compact and
    diffuse bases partition.
    '''
    if precision is None:
        precision = cell.precision * 1e-1

    exps, cs = extract_pgto_params(cell, 'diffuse')
    ls = cell._bas[:,gto.ANG_OF]

    # The most diffuse shell
    r2_approx = np.log(cs**2/precision * 10**ls + 1e-200) / exps
    ai_idx = ak_idx = r2_approx.argmax()

    logger.debug2(cell, 'ai_idx=%d ak_idx=%d', ai_idx, ak_idx)
    ak = exps[ak_idx]
    lk = ls[ak_idx]
    ck = cs[ak_idx]
    aj = exps
    lj = ls
    cj = cs
    ai = exps[ai_idx]
    li = ls[ai_idx]
    ci = cs[ai_idx]
    exp_min_idx = ak_idx
    al = exps[exp_min_idx]
    ll = ls[exp_min_idx]
    cl = cs[exp_min_idx]

    aij = ai + aj
    akl = ak + al
    lij = li + lj
    lkl = lk + ll
    l4 = lij + lkl
    norm_ang = ((2*li+1)*(2*lj+1)*(2*lk+1)*(2*ll+1)/(4*np.pi)**4)**.5
    c1 = ci * cj * ck * cl * norm_ang
    theta = omega**2*aij*akl/(aij*akl + (aij+akl)*omega**2)
    sfac = omega**2*aj*al/(aj*al + (aj+al)*omega**2) / theta
    fl = 2
    fac = 2**(li+lk)*np.pi**2.5*c1 * theta**(l4-.5)
    fac *= 2*np.pi/cell.vol/theta
    fac /= aij**(li+1.5) * akl**(lk+1.5) * aj**lj * al**ll
    fac *= fl / precision

    r0 = cell.rcut
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-1) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-1) + 1.) / (sfac*theta))**.5
    rcut = r0
    return rcut

def _make_tril_pair_mappings(supmol, l_ctr_bas_loc, q_cond, cutoff, tile=4):
    nimgs = len(supmol.Ls)
    cell = supmol.cell
    nbas_cell0 = cell.nbas
    # l_ctr_bas_loc stores the offsets for each l-ctr pattern for the first image.
    # The same pattern can be applied to the remaining images within the supmol.
    # bas_idx_lookup stores the non-negligible shells in supmol for each l-ctr pattern
    bas_mask = np.zeros(nimgs*nbas_cell0, dtype=bool)
    bas_mask[supmol.bas_mask_idx] = True
    bas_mask = bas_mask.reshape(nimgs, nbas_cell0)
    raw_bas_idx = np.empty(nimgs*nbas_cell0, dtype=np.int32)
    raw_bas_idx[supmol.bas_mask_idx] = np.arange(supmol.nbas, dtype=np.int32)
    raw_bas_idx = raw_bas_idx.reshape(nimgs, nbas_cell0)
    n_groups = len(l_ctr_bas_loc) - 1
    bas_idx_lookup = []
    for i in range(n_groups):
        ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
        bas_idx = raw_bas_idx[:,ish0:ish1][bas_mask[:,ish0:ish1]]
        # Align to "tile", padding -1 at the end
        pad_len = (tile*len(bas_idx) - len(bas_idx)) % tile
        bas_idx = np.append(bas_idx, np.full(pad_len, -1, dtype=np.int32))
        bas_idx_lookup.append(asarray(bas_idx.reshape(-1, tile)))

    nbas = q_cond.shape[0]
    q_cond = q_cond.ravel()
    pair_kl_mappings = {}
    pair_ij_mappings = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish = bas_idx_lookup[i][:,None,:,None]
            jsh = bas_idx_lookup[j][None,:,None,:]
            pair_kl = ish * nbas + jsh
            mask = (ish >= 0) & (jsh >= 0)
            pair_ij = pair_kl[mask & (ish < nbas_cell0)]
            pair_ij = pair_ij[q_cond[pair_ij] > cutoff]
            if i == j:
                pair_kl = pair_kl[mask & (ish >= jsh)]
            else:
                pair_kl = pair_kl[mask]
            pair_kl = pair_kl[q_cond[pair_kl] > cutoff]
            pair_ij_mappings[i,j] = asarray(pair_ij, dtype=np.int32)
            pair_kl_mappings[i,j] = asarray(pair_kl, dtype=np.int32)
    return pair_ij_mappings, pair_kl_mappings

def _compressed_dm_cond(supmol, dms):
    '''Largest density matrix elements for each shell-pair. The input and output
    are the abstract arrays that are compressed over the double-lattice-sum
    '''
    cell = supmol.cell
    ao_loc = asarray(cell.ao_loc)
    n_dm, n_Ts, nao = dms.shape[:3]
    i_loc = cp.arange(0, n_Ts*nao, nao, dtype=np.int32)[:,None] + ao_loc[:-1]
    i_loc = cp.append(i_loc.ravel(), np.int32(n_Ts*nao))
    dm_cond = condense('absmax', dms.reshape(n_dm, n_Ts*nao, nao), i_loc, ao_loc)

    bas_mask_idx = supmol.bas_mask_idx
    img_idx, ish_cell0 = divmod(bas_mask_idx, cell.nbas)
    T_in_pair = supmol.Ts_ji_lookup[img_idx[:,None] * n_Ts + img_idx]
    dm_cond = dm_cond[T_in_pair, ish_cell0[:,None], ish_cell0]
    return dm_cond
