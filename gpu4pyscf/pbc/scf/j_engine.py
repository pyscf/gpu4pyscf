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
from pyscf import lib
from pyscf.gto import ANG_OF, gto_norm
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.scf.rsjk import estimate_ke_cutoff_for_omega
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    condense, transpose_sum, contract, asarray)
from gpu4pyscf.gto.mole import group_basis, extract_pgto_params
from gpu4pyscf.scf.jk import (
    libvhf_rys, _vhf, RysIntEnvVars, _scale_sp_ctr_coeff, _nearest_power2,
    apply_coeff_C_mat_CT, apply_coeff_CT_mat_C)
from gpu4pyscf.scf.j_engine import (
    libvhf_md, _make_tile_max_hierarchy, _to_primitive_bas, THREADS, SHM_SIZE, LMAX)
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.pbc.scf.rsjk import ExtendedMole, PBCJKMatrixOpt, OMEGA, _filter_q_cond
from gpu4pyscf.pbc.df import aft

__all__ = [
    'get_j',
]

libvhf_md.PBC_build_j.restype = ctypes.c_int

def get_j(cell, dm, hermi=0, kpts=None, kpts_band=None, vhfopt=None,
          verbose=None):
    '''Compute K matrix
    '''
    if vhfopt is None:
        vhfopt = PBCJMatrixOpt(cell)
    else:
        assert isinstance(vhfopt, PBCJMatrixOpt)
    if vhfopt.supmol is None:
        vhfopt.build(verbose=verbose)
    vj = vhfopt._get_j_sr(dm, hermi, kpts, kpts_band, verbose=verbose)
    vj += vhfopt._get_j_lr(dm, hermi, kpts, kpts_band, verbose=verbose)
    return vj

class PBCJMatrixOpt:

    def __init__(self, cell, omega=None):
        self.cell = cell
        self.verbose = cell.verbose
        self.stdout = cell.stdout

        self.omega = omega
        self.mesh = None
        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.supmol = None

        # Hold cache on GPU devices
        self._rys_envs = {}
        self._q_cond = {}
        self._s_estimator = {}

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('_rys_envs', '_q_cond', '_s_estimator'))

    def build(self, group_size=None, verbose=None):
        assert group_size is None
        log = logger.new_logger(self, verbose)
        cput0 = log.init_timer()
        cell = self.cell
        if self.omega is None or self.omega == 0:
            # TODO: dynamically determine omega based on rcut
            self.omega = OMEGA
        if self.mesh is None:
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)

        cell, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = group_basis(
            cell, 1, group_size, sparse_coeff=True)
        cell.omega = -self.omega
        self.sorted_cell = cell
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = np.asarray(l_ctr_pad_counts, dtype=np.int32)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        prim_cell, self.prim_to_ctr_mapping = _to_primitive_bas(cell)
        self.prim_cell = prim_cell
        # FIXME: should the supmol be regrouped based on l?
        supmol = self.supmol = ExtendedMole.from_cell(prim_cell, self.omega)

        lmax = uniq_l_ctr[:,0].max()
        if lmax > LMAX:
            raise NotImplementedError('basis set with h functions')

        # TODO: approx with overlap mask
        nbas = supmol.nbas
        ao_loc = supmol.ao_loc
        q_cond = np.empty((nbas,nbas))
        intor = supmol._add_suffix('int2e')
        with supmol.with_integral_screen(cell.precision**2*1e-4):
            _vhf.libcvhf.CVHFnr_int2e_q_cond(
                getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
                q_cond.ctypes, ao_loc.ctypes,
                supmol._atm.ctypes, ctypes.c_int(supmol.natm),
                supmol._bas.ctypes, ctypes.c_int(supmol.nbas), supmol._env.ctypes)
        q_cond = np.log(q_cond + 1e-300).astype(np.float32)
        self.q_cond_cpu = q_cond

        diffuse_exps = np.hstack(supmol.bas_exps(), dtype=np.float32)
        diffuse_ctr_coef = gto_norm(supmol._bas[:,ANG_OF], diffuse_exps)
        diffuse_ctr_coef = diffuse_ctr_coef.astype(np.float32)
        s_estimator = np.empty((nbas+2,nbas), dtype=np.float32)
        s_estimator[nbas] = diffuse_exps
        s_estimator[nbas+1] = diffuse_ctr_coef
        libvhf_rys.sr_eri_s_estimator(
            s_estimator.ctypes, ctypes.c_float(supmol.omega),
            diffuse_exps.ctypes, diffuse_ctr_coef.ctypes,
            supmol._atm.ctypes, ctypes.c_int(supmol.natm),
            supmol._bas.ctypes, ctypes.c_int(supmol.nbas), supmol._env.ctypes)
        self.q_cond_cpu = _filter_q_cond(
            supmol, q_cond, s_estimator, self.rys_envs,
            self.estimate_cutoff_with_penalty())[0]
        log.timer('Initialize q_cond', *cput0)
        return self

    def reset(self, cell):
        self.cell = cell
        self.supmol = None
        self._rys_envs = {}
        self._q_cond = {}

    @multi_gpu.property(cache='_q_cond')
    def q_cond(self):
        return asarray(self.q_cond_cpu)

    @multi_gpu.property(cache='_rys_envs')
    def rys_envs(self):
        supmol = self.supmol
        atm = asarray(supmol._atm)
        bas = asarray(supmol._bas)
        env = asarray(_scale_sp_ctr_coeff(supmol))
        ao_loc = asarray(supmol.ao_loc)
        return RysIntEnvVars.new(supmol.natm, supmol.nbas, atm, bas, env, ao_loc)

    estimate_cutoff_with_penalty = PBCJKMatrixOpt.estimate_cutoff_with_penalty

    def _get_j_sr(self, dm, hermi, kpts=None, kpts_band=None, verbose=None):
        '''
        Build K for the sorted_mol over the sampled k-points.
        Return a (*, nkpts, nao, nao) array.

        If the "kpts" is supplied as None or [[0,0,0]] (the gamma point), the K
        matrix is still evaluated as the k-point sampling case. The "nkpts"
        dimension is set to 1
        '''
        log = logger.new_logger(self, verbose)
        cell = self.cell
        assert cell.dimension == 3
        sorted_cell = self.sorted_cell
        nao_orig = cell.nao
        nao = sorted_cell.nao
        supmol = self.supmol
        prim_cell = supmol.cell
        assert supmol.nbas < 65536
        nbas_cell0 = prim_cell.nbas

        dm = asarray(dm, order='C')
        dms = dm.reshape(-1,nao_orig,nao_orig)
        #:dms = cp.einsum('pi,nij,qj->npq', self.coeff, dms, self.coeff)
        dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, self.uniq_l_ctr,
                                   self.l_ctr_offsets, self.ao_idx)
        if hermi != 1:
            dms = transpose_sum(dms)
            dms *= .5

        p2c_mapping = asarray(self.prim_to_ctr_mapping)
        if kpts is None:
            kpts = np.zeros((1, 3))
        else:
            kpts = kpts.reshape(-1, 3)
        is_gamma_point = is_zero(kpts)
        if is_gamma_point:
            assert dms.dtype == np.float64
            nkpts = 1
            ao_loc = asarray(sorted_cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
            log_max_dm = float(dm_cond.max())
            ish_cell0 = supmol.bas_mask_idx % nbas_cell0
            ctr_shell_in_cell0 = p2c_mapping[ish_cell0]
            dm_cond = dm_cond[ctr_shell_in_cell0[:,None], ctr_shell_in_cell0]
        else:
            scaled_kpts = kpts.dot(cell.lattice_vectors().T)
            Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
            expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
            nkpts = expLk.shape[1]
            dms = dms.reshape(-1, nkpts, nao, nao)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            assert abs(dms.imag).max() < 1e-6
            dms = dms.real
            dms = cp.asarray(dms, order='C')
            dm_cond = _dm_cond_from_compressed_dm(supmol, dms, sorted_cell, p2c_mapping)
        n_dm = len(dms)
        log_max_dm = float(dm_cond.max().get())
        log_cutoff = math.log(self.estimate_cutoff_with_penalty())
        q_cutoff = log_cutoff - log_max_dm

        # dm_xyz tensor is compressed over the image-Id dimension. While the
        # tril part of the DM for supmol is required, certain tril part could
        # contribute to the triu part of the compressed dm_xyz. Therefore, all
        # AO-pairs should be transformed.
        ls = prim_cell._bas[:,ANG_OF]
        ll = ls[:,None] + ls
        xyz_size = (ll+1)*(ll+2)*(ll+3)//6
        pair_cum_cell0 = np.cumsum(xyz_size.ravel(), dtype=np.int32)
        pair_loc_in_cell0 = np.append(np.int32(0), pair_cum_cell0)
        dm_xyz_size = pair_cum_cell0[-1]
        log.debug1('dm_xyz_size = %s, nao = %s', dm_xyz_size, nao) # for one image
        nimgs_uniq_pair = len(supmol.double_latsum_Ts)
        npairs = xyz_size.size
        pair_loc = np.arange(0, npairs*nimgs_uniq_pair, npairs,
                             dtype=np.int32)[:,None] + pair_cum_cell0
        pair_loc = np.append(np.int32(0), pair_loc.ravel())
        dms = dms.get()
        dm_xyz = np.empty((n_dm, nimgs_uniq_pair, dm_xyz_size))
        # Must use this modified _env to ensure the consistency with GPU kernel
        # In this _env, normalization coefficients for s and p funcitons are scaled.
        prim_cell_env = _scale_sp_ctr_coeff(prim_cell)
        ao_loc = sorted_cell.ao_loc
        double_latsum_Ls = cp.asnumpy(supmol.double_latsum_Ts).dot(cell.lattice_vectors())
        libvhf_md.PBC_Et_dot_dm(
            dm_xyz.ctypes, dms.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc_in_cell0.ctypes,
            self.prim_to_ctr_mapping.ctypes,
            double_latsum_Ls.ctypes,
            ctypes.c_int(nimgs_uniq_pair),
            ctypes.c_int(int(is_gamma_point)),
            ctypes.c_int(prim_cell.nbas), ctypes.c_int(sorted_cell.nbas),
            prim_cell._bas.ctypes, prim_cell_env.ctypes)

        l_counts = np.bincount(prim_cell._bas[:,ANG_OF])[:LMAX+1]
        n_groups = len(l_counts)
        l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
        l_symb = lib.param.ANGULAR
        pair_ij_mappings, pair_kl_mappings = _make_pair_qd_cond(
            supmol, l_ctr_bas_loc, self.q_cond, dm_cond, q_cutoff,
            pair_loc_in_cell0)
        dm_cond = None

        # TODO: 8-fold symmetry
        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(n_groups)
                 for l in range(k+1))

        def proc(dm_xyz):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(self, verbose)
            t0 = log.init_timer()
            dm_xyz = asarray(dm_xyz) # transfer to current device
            vj_xyz = cp.zeros_like(dm_xyz)

            _pair_ij_mappings = pair_ij_mappings
            _pair_kl_mappings = pair_kl_mappings
            if device_id > 0:
                # Ensure the precomputation avail on each device
                _pair_ij_mappings = {k: [cp.asarray(x) for x in v]
                                     for k, v in pair_ij_mappings.items()}
                _pair_kl_mappings = {k: [cp.asarray(x) for x in v]
                                     for k, v in pair_kl_mappings.items()}
            q_cond = cp.asarray(self.q_cond)
            t1 = log.timer_debug1(f'q_cond on Device {device_id}', *t0)

            timing_counter = Counter()
            kern_counts = 0
            kern = libvhf_md.PBC_build_j
            rys_envs = self.rys_envs

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, pair_ij_loc, qd_ij = _pair_ij_mappings[i,j]
                pair_kl_mapping, pair_kl_loc, qd_kl = _pair_kl_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = _md_j_engine_quartets_scheme(task)
                err = kern(
                    ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm),
                    ctypes.c_int(dm_xyz_size),
                    ctypes.c_int(nimgs_uniq_pair),
                    ctypes.byref(rys_envs), (ctypes.c_int*6)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    supmol._atm.ctypes, ctypes.c_int(supmol.natm),
                    supmol._bas.ctypes, ctypes.c_int(supmol.nbas),
                    supmol._env.ctypes)
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'PBC_build_j kernel for {llll} failed')

                if log.verbose >= logger.DEBUG1:
                    ntasks = pair_ij_mapping.size * pair_kl_mapping.size
                    t1, t1p = log.timer_debug1(f'processing {llll}, scheme={scheme} tasks ~= {ntasks}', *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dm_xyz,), non_blocking=True)

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

        if kpts_band is not None:
            raise NotImplementedError

        vj_xyz = multi_gpu.array_reduce(vj_dist, inplace=True)
        vj_xyz = vj_xyz.get()
        vj, dms = dms, None
        vj[:] = 0.
        assert vj_xyz.ndim == 3
        libvhf_md.PBC_jengine_dot_Et(
            vj.ctypes, vj_xyz.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc_in_cell0.ctypes,
            self.prim_to_ctr_mapping.ctypes,
            double_latsum_Ls.ctypes,
            ctypes.c_int(nimgs_uniq_pair),
            ctypes.c_int(int(is_gamma_point)),
            ctypes.c_int(prim_cell.nbas), ctypes.c_int(sorted_cell.nbas),
            prim_cell._bas.ctypes, prim_cell_env.ctypes)

        if not is_gamma_point:
            expLkz = expLk.view(np.float64).reshape(nimgs_uniq_pair, nkpts, 2)
            vj = vj.reshape(n_dm, nimgs_uniq_pair, nao, nao)
            vj = contract('sLmn,Lkz->skmnz', vj, expLkz)
            vj = cp.asarray(vj, order='C').view(np.complex128)[:,:,:,:,0]
            vj = vj.reshape(-1, nao, nao)

        assert vj.ndim == 3
        vj = transpose_sum(asarray(vj))
        vj *= 2 # because build_j only contracts the tril dm

        vj = apply_coeff_CT_mat_C(vj, cell, sorted_cell, self.uniq_l_ctr,
                                  self.l_ctr_offsets, self.ao_idx)
        if not is_gamma_point:
            weight = 1. / nkpts
            vj *= weight
        if kpts_band is None:
            vj = vj.reshape(dm.shape)
        else:
            raise NotImplementedError
        return vj

    def _get_j_lr(self, dm, hermi, kpts=None, kpts_band=None, verbose=None):
        from gpu4pyscf.pbc.df.aft_jk import get_j_kpts
        cell = self.cell
        assert cell.dimension == 3
        return get_j_kpts(self, dm, hermi, kpts, kpts_band)

    def weighted_coulG(self, kpt=None, exx=None, mesh=None, omega=None, kpts=None):
        '''weighted LR Coulomb kernel. Mimic AFTDF.weighted_coulG'''
        if mesh is None:
            mesh = self.mesh
        cell = self.cell
        omega = self.omega
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = get_coulG(cell, kpt, mesh=mesh, Gv=Gv, wrap_around=True, omega=omega)
        if kpt is None or is_zero(kpt):
            coulG[0] -= np.pi / omega**2
        coulG *= kws
        return coulG

    ft_loop = aft.AFTDF.ft_loop

def _dm_cond_from_compressed_dm(supmol, dms, cell, p2c_mapping):
    '''Largest density matrix elements for each shell-pair. The input and output
    are the abstract arrays that are compressed over the double-lattice-sum
    '''
    prim_cell = supmol.cell
    ao_loc = asarray(cell.ao_loc)
    n_dm, n_Ts, nao = dms.shape[:3]
    Ts_ao_loc = cp.arange(0, n_Ts*nao, nao, dtype=np.int32)[:,None] + ao_loc[:-1]
    Ts_ao_loc = cp.append(Ts_ao_loc.ravel(), np.int32(n_Ts*nao))
    dm_cond = condense('absmax', dms.reshape(n_dm, n_Ts*nao, nao), Ts_ao_loc, ao_loc)
    dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
    nbas = cell.nbas
    dm_cond = dm_cond.reshape(n_Ts, nbas, nbas)
    dm_cond = dm_cond[:,p2c_mapping[:,None], p2c_mapping]

    nbas = prim_cell.nbas
    img_idx, ish_cell0 = divmod(cp.asarray(supmol.bas_mask_idx), nbas)
    # Note the transpose for T_in_pair. dms is stored as [T, ket, bra]
    T_in_pair = cp.asarray(supmol.Ts_ji_lookup)[img_idx,img_idx[:,None]]
    dm_cond = dm_cond[T_in_pair, ish_cell0[:,None], ish_cell0]
    return dm_cond

def _make_pair_qd_cond(supmol, l_ctr_bas_loc, q_cond, dm_cond, cutoff,
                       pair_loc_in_cell0):
    nimgs = len(supmol.Ls)
    cell = supmol.cell
    nbas_cell0 = cell.nbas
    # l_ctr_bas_loc stores the offsets for each l-ctr pattern for the first image.
    # The same pattern can be applied to the remaining images within the supmol.
    # bas_idx_lookup stores the non-negligible shells in supmol for each l-ctr pattern
    bas_mask = cp.zeros(nimgs*nbas_cell0, dtype=bool)
    bas_mask[supmol.bas_mask_idx] = True
    bas_mask = bas_mask.reshape(nimgs, nbas_cell0)
    raw_bas_idx = cp.empty(nimgs*nbas_cell0, dtype=np.uint32)
    raw_bas_idx[supmol.bas_mask_idx] = cp.arange(supmol.nbas, dtype=np.uint32)
    raw_bas_idx = raw_bas_idx.reshape(nimgs, nbas_cell0)
    bas_mask_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.uint32)
    img_idx, sh_cell0 = divmod(bas_mask_idx, nbas_cell0)
    n_groups = len(l_ctr_bas_loc) - 1
    bas_idx_lookup = []
    for i in range(n_groups):
        ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
        bas_idx = asarray(raw_bas_idx[:,ish0:ish1][bas_mask[:,ish0:ish1]])
        bas_idx_lookup.append([bas_idx, img_idx[bas_idx], sh_cell0[bas_idx]])

    pair_loc_in_cell0 = cp.asarray(pair_loc_in_cell0, dtype=np.int32)
    dm_xyz_size = pair_loc_in_cell0[-1]
    Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup, dtype=np.int32)
    q_cond = q_cond.ravel()
    dm_cond = dm_cond.ravel()
    nbas = np.uint32(supmol.nbas)
    pair_ij_mappings = {}
    pair_kl_mappings = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish, iL, ish_cell0 = bas_idx_lookup[i]
            jsh, jL, jsh_cell0 = bas_idx_lookup[j]
            pair_idx = ish[:,None] * nbas + jsh
            if i == j:
                # pair_ij includes only the shell i within the first image.
                pair_ij = pair_idx[(iL[:,None] == 0) & (ish_cell0[:,None] >= jsh_cell0)]
                pair_kl = pair_idx[ish[:,None] >= jsh]
            else:
                pair_ij = pair_idx[iL == 0].ravel()
                pair_kl = pair_idx.ravel()
            pair_ij = cp.asarray(pair_ij[q_cond[pair_ij] > cutoff], dtype=np.uint32)
            pair_kl = cp.asarray(pair_kl[q_cond[pair_kl] > cutoff], dtype=np.uint32)
            pair_ij = pair_ij[cp.argsort(q_cond[pair_ij])[::-1]]
            pair_kl = pair_kl[cp.argsort(q_cond[pair_kl])[::-1]]

            bas_i, bas_j = divmod(pair_ij, nbas)
            bas_k, bas_l = divmod(pair_kl, nbas)
            iL = img_idx[bas_i]
            jL = img_idx[bas_j]
            kL = img_idx[bas_k]
            lL = img_idx[bas_l]
            ish_cell0 = sh_cell0[bas_i]
            jsh_cell0 = sh_cell0[bas_j]
            ksh_cell0 = sh_cell0[bas_k]
            lsh_cell0 = sh_cell0[bas_l]
            ij_loc = pair_loc_in_cell0[ish_cell0*nbas_cell0+jsh_cell0]
            ij_loc += Ts_ji_lookup[iL, jL] * dm_xyz_size
            kl_loc = pair_loc_in_cell0[ksh_cell0*nbas_cell0+lsh_cell0]
            kl_loc += Ts_ji_lookup[kL, lL] * dm_xyz_size

            # qd_tile_max is the product of q_cond and dm_cond within each batch
            qd_ij = q_cond[pair_ij] + dm_cond[pair_ij]
            qd_kl = q_cond[pair_kl] + dm_cond[pair_kl]
            pair_ij_mappings[i,j] = (pair_ij, ij_loc, _make_tile_max_hierarchy(qd_ij))
            pair_kl_mappings[i,j] = (pair_kl, kl_loc, _make_tile_max_hierarchy(qd_kl))
    return pair_ij_mappings, pair_kl_mappings

VJ_IJ_REGISTERS = 11
RT_TMP_REGISTERS = 31
RT2_IDX_CACHE_SIZE = 35 * 56
def _md_j_engine_quartets_scheme(ls, shm_size=SHM_SIZE):
    n_dm = 1
    vj_ij_registers = VJ_IJ_REGISTERS
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nf3ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt_size = (order+1)*(order+2)*(order+3)//6
    gout_stride_min = max(
        _nearest_power2(int((nf3ij+vj_ij_registers-1) / vj_ij_registers), False),
        _nearest_power2(int((Rt_size+RT_TMP_REGISTERS-1) / RT_TMP_REGISTERS), False))

    unit = order+1 + Rt_size
    #counts = shm_size // ((unit+gout_stride_min-1)//gout_stride_min*8)
    counts = shm_size // (unit*8)
    threads = THREADS
    if counts * gout_stride_min >= threads:
        nsq = threads // gout_stride_min
    else:
        nsq = _nearest_power2(counts)
    kl = _nearest_power2(int(nsq**.5))
    ij = nsq // kl

    cache_Rt2_idx = nf3ij * nf3kl <= RT2_IDX_CACHE_SIZE
    if cache_Rt2_idx:
        shm_size -= nf3ij * nf3kl * 2

    tilex = 32
    # Guess number of batches for kl indices
    tiley = (shm_size//8 - nsq*unit - ij*4) // (kl*4+kl*nf3kl*n_dm)
    tiley = min(tilex, tiley)
    tiley = tiley // 4 * 4
    if tiley < 4:
        tiley = 4
    if li == lk and lj == ll:
        tilex = tiley
    cache_size = ij * 4 + kl*tiley * 4 + kl*nf3kl*tiley*n_dm
    while (nsq * unit + cache_size) * 8 > shm_size:
        nsq //= 2
        assert nsq >= 1
        kl = _nearest_power2(int(nsq**.5))
        ij = nsq // kl
        cache_size = ij * 4 + kl*tiley * 4 + kl*nf3kl*tiley*n_dm
    gout_stride = threads // nsq
    buflen = (nsq * unit + cache_size) * 8
    if cache_Rt2_idx:
        buflen += nf3ij * nf3kl * 2
    return ij, kl, gout_stride, tilex, tiley, buflen
