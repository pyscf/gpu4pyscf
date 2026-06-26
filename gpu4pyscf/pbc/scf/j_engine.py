# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto
from pyscf.gto import ANG_OF, gto_norm
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.scf.rsjk import estimate_ke_cutoff_for_omega
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    condense, transpose_sum, contract, asarray, ndarray)
from gpu4pyscf.gto.mole import groupby, extract_pgto_params, SortedCell
from gpu4pyscf.scf.jk import (
    libvhf_rys, _vhf, RysIntEnvVars, _scale_sp_ctr_coeff, _nearest_power2,
    _TimingCollector)
from gpu4pyscf.scf.j_engine import (
    libvhf_md, _make_tile_max_hierarchy, THREADS, SHM_SIZE, LMAX)
from gpu4pyscf.pbc.df.fft import _check_kpts
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.pbc.scf.rsjk import (
    NBAS_MAX, OMEGA, libpbc, ExtendedMole, PBCJKMatrixOpt)
from gpu4pyscf.pbc.df import aft

__all__ = [
    'get_j',
]

libvhf_md.PBC_build_j.restype = ctypes.c_int
libvhf_md.init_mdj_constant.restype = ctypes.c_int

def get_j(cell, dm, hermi=0, kpts=None, kpts_band=None, vhfopt=None,
          verbose=None):
    '''Compute J matrix
    '''
    if vhfopt is None:
        vhfopt = PBCJMatrixOpt(cell)
    else:
        assert isinstance(vhfopt, PBCJMatrixOpt)
    return vhfopt.get_j(dm, hermi, kpts, kpts_band)

class PBCJMatrixOpt:

    def __init__(self, cell, omega=None):
        self.cell = cell
        self.verbose = cell.verbose
        self.stdout = cell.stdout

        self.omega = omega
        self.mesh = None
        self.supmol = None

        # Hold cache on GPU devices
        self._rys_envs = {}

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('_rys_envs', '_q_cond', '_s_estimator'))

    def build(self, kpts=None, verbose=None):
        from gpu4pyscf.pbc.dft.multigrid_v2 import _unique_image_pair
        log = logger.new_logger(self, verbose)
        cput0 = log.init_timer()
        # diffuse_cutoff=1e200 to ensure all basis are decontracted to
        # primitive shells.
        cell = self.cell = SortedCell.from_cell(
            self.cell, decontract=True, diffuse_cutoff=1e200)
        lmax = cell.uniq_l_ctr[:,0].max()
        if lmax > LMAX:
            raise NotImplementedError('basis set with h functions')

        if self.omega is None or self.omega == 0:
            self.omega = OMEGA
        if self.mesh is None:
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)

        cell.omega = -self.omega
        log.debug1('PBCJKMatrixOpt.build: omega = %g mesh = %s', self.omega, self.mesh)

        # FIXME: should the supmol be regrouped based on l?
        supmol = self.supmol = ExtendedMole.from_cell(cell, self.omega)
        nimgs = len(supmol.Ls)
        translation_vectors = asarray(np.linalg.solve(cell.lattice_vectors().T, supmol.Ls.T).T)
        translation_vectors = cp.asarray(translation_vectors.round(), dtype=np.int32)
        supmol.double_latsum_Ts, inverse = _unique_image_pair(translation_vectors)
        supmol.Ts_ji_lookup = cp.asarray(inverse, order='C', dtype=np.int32).reshape(nimgs, nimgs)

        self.bas_pair_cache = _cache_q_cond_and_non0pairs(self)
        log.timer('Initialize q_cond', *cput0)
        return self

    def reset(self, cell):
        self.cell = cell
        self.supmol = None
        self._rys_envs = {}

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
        prim_cell = cell = self.cell
        assert cell.dimension == 3
        nao = cell.nao
        supmol = self.supmol
        assert supmol.nbas < 65536

        dm = asarray(dm, order='C')
        nao_orig = dm.shape[-1]
        dms = cell.apply_C_mat_CT(dm.reshape(-1,nao_orig,nao_orig))
        if hermi != 1:
            dms = transpose_sum(dms)
            dms *= .5

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        is_gamma_point = is_zero(kpts)
        if is_gamma_point:
            if is_single_kpt:
                assert dms.dtype == np.float64
            else:
                dms = dms.real
            nkpts = 1
            ao_loc = asarray(cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            nbas_cell0 = prim_cell.nbas
            ish_cell0 = supmol.bas_mask_idx % nbas_cell0
            dm_cond = dm_cond[ish_cell0[:,None], ish_cell0]
            n_dm = len(dms)
        else:
            scaled_kpts = kpts.dot(cell.lattice_vectors().T)
            Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
            expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
            nkpts = expLk.shape[1]
            dms = dms.reshape(-1, nkpts, nao, nao)
            n_dm = len(dms)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            dms = dms.real
            dms = cp.asarray(dms, order='C')
            dm_cond = _dm_cond_from_compressed_dm(supmol, dms)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_cutoff = math.log(self.estimate_cutoff_with_penalty())

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
        ao_loc = prim_cell.ao_loc
        double_latsum_Ls = cp.asnumpy(supmol.double_latsum_Ts).dot(cell.lattice_vectors())
        libvhf_md.PBC_Et_dot_dm(
            dm_xyz.ctypes, dms.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc_in_cell0.ctypes,
            double_latsum_Ls.ctypes,
            ctypes.c_int(nimgs_uniq_pair),
            ctypes.c_int(int(is_gamma_point)),
            ctypes.c_int(prim_cell.nbas), ctypes.c_int(prim_cell.nbas),
            prim_cell._bas.ctypes, prim_cell_env.ctypes)

        l_ctr_bas_loc = np.append(0, np.cumsum(prim_cell.l_ctr_counts))
        uniq_l = prim_cell.uniq_l_ctr[:,0]
        n_groups = len(l_ctr_bas_loc) - 1
        l_symb = lib.param.ANGULAR
        bas_pair_qd_cache = _make_pair_qd_cond(self, dm_cond, pair_loc_in_cell0)
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
            t1 = log.init_timer()
            dm_xyz = asarray(dm_xyz) # transfer to current device
            vj_xyz = cp.zeros_like(dm_xyz)

            _bas_pair_cache = bas_pair_qd_cache
            if device_id > 0:
                # Ensure the precomputation avail on each device
                _bas_pair_cache = {k: [cp.asarray(x) for x in v]
                                   for k, v in bas_pair_qd_cache.items()}

            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            err = libvhf_md.init_mdj_constant(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError(f'MD-J kernel init failed on Device {device_id}')
            kern = libvhf_md.PBC_build_j
            rys_envs = self.rys_envs

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, pair_ij_loc, qd_ij = _bas_pair_cache[i,j][:4]
                pair_kl_mapping, q_cond_kl, pair_kl_loc, qd_kl = _bas_pair_cache[k,l][4:]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                ls = uniq_l[list(task)]
                scheme = _md_j_engine_quartets_scheme(ls)
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
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    supmol._atm.ctypes, ctypes.c_int(supmol.natm),
                    supmol._bas.ctypes, ctypes.c_int(supmol.nbas),
                    supmol._env.ctypes)
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'PBC_build_j kernel for {llll} failed')
                kern_counts += 1

                if log.verbose >= logger.DEBUG1:
                    ntasks = pair_ij_mapping.size * pair_kl_mapping.size
                    msg = f'processing {llll}, scheme={scheme} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dm_xyz,), non_blocking=True)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            _TimingCollector.summary(log.debug1, (x[2] for x in results))

        if kpts_band is not None:
            raise NotImplementedError

        vj_xyz = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        vj_xyz = vj_xyz.get()
        vj, dms = dms, None
        vj[:] = 0.
        assert vj_xyz.ndim == 3
        libvhf_md.PBC_jengine_dot_Et(
            vj.ctypes, vj_xyz.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc_in_cell0.ctypes,
            double_latsum_Ls.ctypes,
            ctypes.c_int(nimgs_uniq_pair),
            ctypes.c_int(int(is_gamma_point)),
            ctypes.c_int(prim_cell.nbas), ctypes.c_int(prim_cell.nbas),
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

        vj = prim_cell.apply_CT_mat_C(vj)
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

    def get_j(self, dm, hermi=0, kpts=None, kpts_band=None):
        '''Compute J matrix
        '''
        if self.supmol is None:
            self.build()
        vj = self._get_j_sr(dm, hermi, kpts, kpts_band)
        vj += self._get_j_lr(dm, hermi, kpts, kpts_band)
        return vj

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

def _dm_cond_from_compressed_dm(supmol, dms):
    '''Largest density matrix elements for each shell-pair. The input and output
    are the abstract arrays that are compressed over the double-lattice-sum
    '''
    prim_cell = cell = supmol.cell
    ao_loc = asarray(cell.ao_loc)
    n_dm, n_Ts, nao = dms.shape[:3]
    Ts_ao_loc = cp.arange(0, n_Ts*nao, nao, dtype=np.int32)[:,None] + ao_loc[:-1]
    Ts_ao_loc = cp.append(Ts_ao_loc.ravel(), np.int32(n_Ts*nao))
    dm_cond = condense('absmax', dms.reshape(n_dm, n_Ts*nao, nao), Ts_ao_loc, ao_loc)
    dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
    nbas = cell.nbas
    dm_cond = dm_cond.reshape(n_Ts, nbas, nbas)

    nbas = prim_cell.nbas
    img_idx, ish_cell0 = divmod(cp.asarray(supmol.bas_mask_idx), nbas)
    # Note the transpose for T_in_pair. dms is stored as [T, ket, bra]
    T_in_pair = cp.asarray(supmol.Ts_ji_lookup)[img_idx,img_idx[:,None]]
    dm_cond = dm_cond[T_in_pair, ish_cell0[:,None], ish_cell0]
    return dm_cond

def _cache_q_cond_and_non0pairs(vhfopt):
    # This is similar to rsjk._cache_q_cond_and_non0pairs. The difference is the
    # pair_kl to adapt to the 4-fold symmetry in current MD implementation.
    cell = vhfopt.cell
    supmol = vhfopt.supmol
    omega = -vhfopt.omega

    precision = vhfopt.estimate_cutoff_with_penalty()
    # Adjust precision to improve accuracy for very diffuse orbitals
    s_log_cutoff = q_log_cutoff = math.log(precision)

    diffuse_exps, diffuse_ctr_coef = extract_pgto_params(cell, 'diffuse')
    diffuse_idx = groupby(cell._bas[:,gto.ATOM_OF], diffuse_exps, 'argmin')
    diffuse_exps_per_atom = cp.array(diffuse_exps[diffuse_idx], dtype=np.float32)
    diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
    diffuse_ctr_coef = cp.asarray(diffuse_ctr_coef, dtype=np.float32)

    SIZEOF_FLOAT = ctypes.sizeof(ctypes.c_float)
    gout_width = 29
    ls = np.arange(LMAX+1)
    li = ls[:,None]
    lj = ls
    lij = li + lj
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nroots = lij + 1
    nroots *= 2 # for SR integrals
    unit = (li+1)*(lj+1)*2 + (li+1)*(lj+1)*(lij+1) + 6 + nroots*4
    shm_size = 1024 * 48 - 1024
    nsp_max = _nearest_power2(shm_size // (unit*SIZEOF_FLOAT))
    gout_size = nfi * nfj
    gout_stride = (gout_size+gout_width-1) // gout_width
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    nsp_per_block = THREADS // gout_stride
    # min(nsp_per_block, nsp_max)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit * SIZEOF_FLOAT)
    # (pp|pp) requires more shm than this estimation. 5888 is the required size
    max_shm_size = max(shm_size.max(), 5888*SIZEOF_FLOAT)

    cell = supmol.cell
    nbas_cell0 = cell.nbas
    nbas = supmol.nbas
    nimgs = len(supmol.Ls)
    l_ctr_bas_loc = np.append(0, np.cumsum(cell.l_ctr_counts))
    # l_ctr_bas_loc stores the offsets for each l-ctr pattern for the first image.
    # The same pattern can be applied to the remaining images within the supmol.
    # bas_idx_lookup stores the non-negligible shells in supmol for each l-ctr pattern
    bas_mask = cp.zeros(nimgs*nbas_cell0, dtype=bool)
    bas_mask_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.int32)
    bas_mask[bas_mask_idx] = True
    bas_mask = bas_mask.reshape(nimgs, nbas_cell0)

    raw_bas_idx = cp.empty(nimgs*nbas_cell0, dtype=np.int64)
    raw_bas_idx[bas_mask_idx] = cp.arange(nbas, dtype=np.int64)
    raw_bas_idx = raw_bas_idx.reshape(nimgs, nbas_cell0)
    n_groups = len(l_ctr_bas_loc) - 1
    bas_idx_lookup = []
    for i in range(n_groups):
        ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
        bas_idx = raw_bas_idx[:,ish0:ish1][bas_mask[:,ish0:ish1]]
        # Be careful with the overflow for "bas_idx[:,None] * NBAS_MAX"
        bas_idx_lookup.append(cp.asarray(bas_idx, dtype=np.int64, order='C'))

    n = max(x.size for x in bas_idx_lookup)
    pair_buf = cp.empty(n**2, dtype=np.int64)
    s_buf = cp.empty(n**2, dtype=np.float32)

    pair_ij_kern = libpbc.PBCsort_pair_ij
    s_kern = libpbc.PBCfill_s_estimator
    q_kern = libpbc.PBCfill_qcond
    pair_ij_kern.restype = ctypes.c_int
    q_kern.restype = ctypes.c_int
    s_kern.restype = ctypes.c_int
    rys_envs = vhfopt.rys_envs
    pair_cache = {}
    for i in range(n_groups):
        for j in range(i+1):
            nish_cell0 = cell.l_ctr_counts[i]
            ish = bas_idx_lookup[i]
            jsh = bas_idx_lookup[j]
            nish = len(ish)
            njsh = len(jsh)
            pair_ij = ndarray((nish_cell0, njsh), dtype=np.int64, buffer=pair_buf)
            cp.add(ish[:nish_cell0,None]*NBAS_MAX, jsh, out=pair_ij)
            pair_ij = pair_ij.ravel()
            tril_symmetry = 1 if i == j else 0
            s_estimator = ndarray(pair_ij.shape, dtype=np.float32, buffer=s_buf)
            err = s_kern(ctypes.cast(s_estimator.data.ptr, ctypes.c_void_p),
                         ctypes.byref(rys_envs),
                         ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
                         ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_exps_per_atom.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_ctr_coef.data.ptr, ctypes.c_void_p),
                         ctypes.c_float(s_log_cutoff),
                         ctypes.c_int(nbas_cell0),
                         ctypes.c_int(len(diffuse_exps_per_atom)),
                         ctypes.c_int(pair_ij.size),
                         ctypes.c_double(omega),
                         ctypes.c_int(tril_symmetry),
                         lib.c_null_ptr())
            if err != 0:
                raise RuntimeError('PBCfill_s_estimator kernel failed')
            idx = cp.where(s_estimator > s_log_cutoff)[0]
            pair_ij = pair_ij[idx]
            s_estimator = s_estimator[idx]

            q_cond_ij = cp.empty(pair_ij.size, dtype=np.float32)
            if len(pair_ij) > 0:
                err = q_kern(ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                             ctypes.byref(rys_envs), ctypes.c_int(max_shm_size),
                             ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
                             ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                             ctypes.c_int(pair_ij.size),
                             ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError('PBCfill_qcond kernel failed')
                idx = cp.where(q_cond_ij > q_log_cutoff)[0]
                s_estimator = s_estimator[idx]
                q_cond_ij = q_cond_ij[idx]
                pair_ij = pair_ij[idx]

            # pair_kl requires ksh>=lsh instead of ksh_cell0>=lsh_cell0
            pair_kl = ndarray((nish, njsh), dtype=np.int64, buffer=pair_buf)
            cp.add(ish[:,None]*NBAS_MAX, jsh, out=pair_kl)
            pair_kl = pair_kl.ravel()
            tril_symmetry = 2 if i == j else 0
            s_estimator = ndarray(pair_kl.shape, dtype=np.float32, buffer=s_buf)
            err = s_kern(ctypes.cast(s_estimator.data.ptr, ctypes.c_void_p),
                         ctypes.byref(rys_envs),
                         ctypes.cast(pair_kl.data.ptr, ctypes.c_void_p),
                         ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_exps_per_atom.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_ctr_coef.data.ptr, ctypes.c_void_p),
                         ctypes.c_float(s_log_cutoff),
                         ctypes.c_int(nbas_cell0),
                         ctypes.c_int(len(diffuse_exps_per_atom)),
                         ctypes.c_int(pair_kl.size),
                         ctypes.c_double(omega),
                         ctypes.c_int(tril_symmetry),
                         lib.c_null_ptr())
            if err != 0:
                raise RuntimeError('PBCfill_s_estimator kernel failed')
            idx = cp.where(s_estimator > s_log_cutoff)[0]
            pair_kl = pair_kl[idx]
            s_estimator = s_estimator[idx]
            q_cond_kl = cp.empty(pair_kl.size, dtype=np.float32)
            if len(pair_kl) > 0:
                err = q_kern(ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                             ctypes.byref(rys_envs), ctypes.c_int(max_shm_size),
                             ctypes.cast(pair_kl.data.ptr, ctypes.c_void_p),
                             ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                             ctypes.c_int(pair_kl.size),
                             ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError('PBCfill_qcond kernel failed')
                idx = cp.where(q_cond_kl > q_log_cutoff)[0]
                s_estimator = s_estimator[idx]
                q_cond_kl = q_cond_kl[idx]
                pair_kl = pair_kl[idx]

            ij_idx = cp.argsort(q_cond_ij)[::-1]
            kl_idx = cp.argsort(q_cond_kl)[::-1]
            pair_cache[i,j] = (pair_ij[ij_idx], q_cond_ij[ij_idx],
                               pair_kl[kl_idx], q_cond_kl[kl_idx])
    return pair_cache

def _make_pair_qd_cond(vhfopt, dm_cond, pair_loc_in_cell0):
    supmol = vhfopt.supmol
    cell = supmol.cell
    nbas = supmol.nbas
    nbas_cell0 = cell.nbas
    bas_mask_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.uint32)
    img_idx, sh_cell0 = divmod(bas_mask_idx, nbas_cell0)

    pair_loc_in_cell0 = cp.asarray(pair_loc_in_cell0, dtype=np.int32)
    dm_xyz_size = pair_loc_in_cell0[-1]
    Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup, dtype=np.int32)
    dm_cond = dm_cond.ravel()

    bas_pair_cache = vhfopt.bas_pair_cache
    bas_pair_qd_cache = {}
    for key in bas_pair_cache:
        pair_ij, q_cond_ij, pair_kl, q_cond_kl = bas_pair_cache[key]
        bas_i, bas_j = divmod(pair_ij, NBAS_MAX)
        bas_k, bas_l = divmod(pair_kl, NBAS_MAX)
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

        qd_ij = q_cond_ij + dm_cond[bas_i*nbas+bas_j]
        qd_kl = q_cond_kl + dm_cond[bas_k*nbas+bas_l]
        bas_pair_qd_cache[key] = (
            pair_ij, q_cond_ij, ij_loc, _make_tile_max_hierarchy(qd_ij),
            pair_kl, q_cond_kl, kl_loc, _make_tile_max_hierarchy(qd_kl))
    return bas_pair_qd_cache

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
