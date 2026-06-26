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
from pyscf.scf import _vhf
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from pyscf.pbc.lib.kpts_helper import is_zero, member
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    condense, transpose_sum, dist_matrix, contract, asarray, ndarray,
    get_avail_mem, absmax)
from gpu4pyscf.gto.mole import (
    groupby, extract_pgto_params, most_diffuse_pgto, SortedCell)
from gpu4pyscf.scf.jk import (
    libvhf_rys, RysIntEnvVars, _scale_sp_ctr_coeff, _nearest_power2,
    _check_rsh_factors, _TimingCollector,
    PTR_BAS_COORD, LMAX, QUEUE_DEPTH, SHM_SIZE, GOUT_WIDTH, THREADS)
from gpu4pyscf.pbc.df.ft_ao import libpbc, FTOpt
from gpu4pyscf.pbc.df.fft import _check_kpts
from gpu4pyscf.pbc.df.fft_jk import _format_dms
from gpu4pyscf.pbc.df import aft, aft_jk
from gpu4pyscf.pbc.tools.k2gamma import (
    kpts_to_kmesh, double_translation_indices)
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter as bvk_kk_adapted_iter
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.pbc.tools.pbc import get_coulG, probe_charge_sr_coulomb
from gpu4pyscf.grad.rhf import _ejk_quartets_scheme
from gpu4pyscf.pbc.gto import int1e

__all__ = [
    'get_k',
]

libpbc.PBC_build_k.restype = ctypes.c_int
libpbc.PBC_build_k_init.restype = ctypes.c_int
libpbc.PBC_build_jk_ip1_init.restype = ctypes.c_int
libpbc.PBC_build_j_init.restype = ctypes.c_int
libpbc.PBC_per_atom_jk_ip1.restype = ctypes.c_int
libpbc.PBC_jk_strain_deriv.restype = ctypes.c_int

DD_CACHE_MAX = 101250 * (SHM_SIZE//48000)
OMEGA = 0.4
NBAS_MAX = 1048576
Q_COND_MARGIN = 4.

def get_k(cell, dm, hermi=0, kpts=None, kpts_band=None, omega=None, vhfopt=None,
          lr_factor=None, sr_factor=None, exxdiv=None, verbose=None):
    '''Compute K matrix
    '''
    omega, lr_factor, sr_factor = _check_rsh_factors(cell, omega, lr_factor, sr_factor)
    omega = abs(omega)

    if vhfopt is None:
        vhfopt = PBCJKMatrixOpt(cell)
    else:
        assert isinstance(vhfopt, PBCJKMatrixOpt)
    if vhfopt.supmol is None:
        if omega is not None and omega != 0 and vhfopt.omega is None:
            rsjk_omega, ke_cutoff, mesh = _guess_omega(cell, kpts)
            logger.debug(cell, 'omega = %g, rsjk omega = %g', omega, rsjk_omega)
            if abs(omega) > rsjk_omega:
                vhfopt.omega = omega
            else:
                vhfopt.omega = rsjk_omega
                vhfopt.mesh = mesh
        vhfopt.build(kpts, verbose=verbose)

    vk = 0
    if sr_factor != 0:
        vk = vhfopt._get_k_sr(dm, hermi, kpts, kpts_band, exxdiv, omega,
                              lr_factor, sr_factor, verbose=verbose)

    if lr_factor != 0 or omega != vhfopt.omega:
        vk += vhfopt._get_k_lr(dm, hermi, kpts, kpts_band, exxdiv, omega,
                               lr_factor, sr_factor)
    return vk

def get_j(cell, dm, hermi=1, kpts=None, kpts_band=None, vhfopt=None):
    '''Compute K matrix
    '''
    if vhfopt is None:
        vhfopt = PBCJKMatrixOpt(cell)
    else:
        assert isinstance(vhfopt, PBCJKMatrixOpt)
    if vhfopt.supmol is None:
        vhfopt.build(kpts)
    vj = vhfopt._get_j_sr(dm, hermi, kpts, kpts_band)
    vj += vhfopt._get_j_lr(dm, hermi, kpts, kpts_band)
    return vj

class PBCJKMatrixOpt:

    def __init__(self, cell, omega=None):
        self.cell = cell
        self.verbose = cell.verbose
        self.stdout = cell.stdout

        self.omega = omega
        self.mesh = None
        self.ke_cutoff = None
        self.supmol = None
        self.exclude_dd_block = True

        # Attributes required by AFTDF functions
        self.time_reversal_symmetry = True

        # Hold cache on GPU devices
        self._rys_envs = {}
        self.dd_bas_idx = None
        self.dd_ao_idx = None

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('_rys_envs', '_q_cond', '_s_estimator'))

    def build(self, kpts=None, verbose=None):
        log = logger.new_logger(self, verbose)
        cput0 = log.init_timer()
        cell = self.cell = SortedCell.from_cell(
            self.cell, decontract=True, diffuse_cutoff=0.3)
        lmax = cell.uniq_l_ctr[:,0].max()
        if lmax > LMAX:
            raise NotImplementedError('basis set with h functions')

        ke_cutoff = mesh = None
        if self.mesh is not None:
            mesh = self.mesh
            ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh)
            ke_cutoff = ke_cutoff[:cell.dimension].min()
        if self.omega is None or self.omega == 0:
            if mesh is None: # None of self.mesh and self.omega are specified
                self.omega, ke_cutoff, self.mesh = _guess_omega(cell.cell, kpts)
            else: # when self.mesh is specified by user
                self.omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff.max())
        if self.mesh is None: # when self.omega is specified by user
            ke_cutoff = estimate_ke_cutoff_for_omega(cell.cell, self.omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
        if self.ke_cutoff is None:
            self.ke_cutoff = ke_cutoff

        cell.omega = -self.omega
        log.debug1('PBCJKMatrixOpt.build: omega = %g mesh = %s ke_cutoff = %g',
                   self.omega, self.mesh, self.ke_cutoff)

        self.supmol = ExtendedMole.from_cell(cell, self.omega)

        pair_mask = None
        if self.exclude_dd_block:
            pair_mask = _search_diffuse_pairs(cell, self.mesh)
            nao = cell.nao
            bas_ij_idx = np.where(pair_mask.ravel().get())[0]
            bas_ij_idx = np.asarray(bas_ij_idx, dtype=np.int32)
            npairs = len(bas_ij_idx)
            ao_loc = cell.ao_loc
            dd_ao_idx = np.empty(nao**2, dtype=np.int32)
            libvhf_rys.ao_pair_indices.restype = ctypes.c_int
            n = libvhf_rys.ao_pair_indices(
                dd_ao_idx.ctypes, bas_ij_idx.ctypes, ao_loc.ctypes,
                ctypes.c_int(npairs), ctypes.c_int(cell.nbas),
                ctypes.c_int(nao))
            self.dd_bas_idx = bas_ij_idx
            self.dd_ao_idx = asarray(dd_ao_idx[:n])
            log.debug1('len(dd_ao_idx) = %d', n)

        self.bas_pair_cache = _cache_q_cond_and_non0pairs(self, 6, pair_mask)
        log.timer('Initialize q_cond', *cput0)
        return self

    def reset(self, cell):
        self.cell = cell
        self.supmol = None
        self._rys_envs = {}

    @multi_gpu.property(cache='_rys_envs')
    def rys_envs(self):
        supmol = self.supmol
        atm = asarray(supmol._atm)
        bas = asarray(supmol._bas)
        env = asarray(_scale_sp_ctr_coeff(supmol))
        ao_loc = asarray(supmol.ao_loc)
        return RysIntEnvVars.new(supmol.natm, supmol.nbas, atm, bas, env, ao_loc)

    def estimate_cutoff_with_penalty(self, precision=None):
        cell = self.cell
        if precision is None:
            precision = cell.precision
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
        # When exp_min is small, the lattice sum over j and k in (ij|kl) would
        # contribute to the kl-pair near the cutoff edges. Accurate estimation
        # for their contributions is hard to derive. Numerical tests show that
        # the contribution is approximately proportional to 1/(exp_min**3*vol**2).
        double_lat_sum_penalty = max(1, 1e6/(exp_min**3*vol**2))
        cutoff = precision / (lattice_sum_factor + double_lat_sum_penalty)
        logger.debug1(cell, 'rsjk integral theta=%g cutoff=%g '
                      'lattice_sum_factor=%g double_lat_sum_penalty=%g',
                      theta, cutoff, lattice_sum_factor, double_lat_sum_penalty)
        return cutoff

    def _get_k_sr(self, dm, hermi, kpts=None, kpts_band=None, exxdiv=None,
                  omega=None, lr_factor=1, sr_factor=1, verbose=None):
        '''
        Build kpts adapted K matrices
        Return a (*, nkpts, nao, nao) array.

        If the "kpts" is supplied as None or [[0,0,0]] (the gamma point), the K
        matrix is still evaluated as the k-point sampling case. The "nkpts"
        dimension is set to 1
        '''
        log = logger.new_logger(self, verbose)
        cell = self.cell
        assert cell.dimension == 3
        nao = cell.nao
        supmol = self.supmol

        dm = asarray(dm)
        nao_orig = dm.shape[-1]
        dms = cell.apply_C_mat_CT(dm.reshape(-1,nao_orig,nao_orig))

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=True)
        # Indicates how the image -I and I in lattice sum are related
        img_conj_mapping = slice(None, None, -1)
        is_gamma_point = is_zero(kpts)
        is_real = True
        if is_gamma_point:
            if is_single_kpt:
                assert dms.dtype == np.float64
            else:
                dms = dms.real
            nkpts = 1
            ao_loc = asarray(cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            if hermi == 0:
                # Wrap the triu contribution to tril
                dm_cond = dm_cond + dm_cond.T
            # Additional dimension for kpts
            dms = dms[:,None,:,:]
            n_dm = len(dms)
            nimgs = nimgs_uniq_pair = 1
            Ts_ji_lookup = cp.zeros((nimgs, nimgs))
            sup_bas_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.int32) % cell.nbas
        else:
            bvk_ncells = np.prod(kmesh)
            nimgs = len(supmol.Ls)
            # When the size of BvK cell is smaller than the supmol, it's more
            # efficient to represent dms/vk in BvK cell
            if bvk_ncells < 7*nimgs:
                sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_bvk(supmol, kmesh, kpts)
                nimgs = bvk_ncells
                img_conj_mapping = conj_images_in_bvk_cell(kmesh)
            else:
                sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_supermol(supmol, kpts)
            nimgs_uniq_pair, nkpts = expLk.shape
            dms = dms.reshape(-1, nkpts, nao, nao)
            n_dm = len(dms)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            if absmax(dms.imag) < cell.precision*5e2:
                dms = dms.real
                dms = cp.asarray(dms, order='C')
            else:
                is_real = False
                dms = cp.vstack([dms.real, dms.imag])
            dm_cond = _dm_cond_from_compressed_dm(supmol, dms)
            if hermi != 1:
                dm_cond = dm_cond + dm_cond.transpose(0,2,1)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_cutoff = math.log(self.estimate_cutoff_with_penalty())
        dm_penalty = float(dm_cond.max())
        log.debug1('dm_penalty = %f', dm_penalty)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(supmol, 'diffuse')

        omega, lr_factor, sr_factor = _check_rsh_factors(cell.cell, omega, lr_factor, sr_factor)
        omega = abs(omega)

        uniq_l_ctr = cell.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(cell.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(self, verbose)
            t1 = log.init_timer()
            dms = cp.asarray(dms)
            dm_cond = cp.asarray(dm_cond)

            if hermi == 0:
                # Contract the tril and triu parts separately
                # Swapping the two orbital indicies also indicates the transpose
                # of Ts_ji_lookup mapping. Ts_ji_lookup.T for double_latsum_Ts
                # happens to be the Ts_ji_lookup for the reversed double_latsum_Ts.
                # Since the same Ts_ji_lookup is applied for both tril and triu,
                # reversing the lattice sum order in triu to accommodate the
                # reversed double_latsum_Ts. The output triu-vk needs to be
                # reversed as well
                dms = cp.vstack([dms, dms[:,img_conj_mapping].transpose(0,1,3,2)])
            dm_counts = len(dms)

            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            _sup_bas_idx = cp.asarray(sup_bas_idx)
            _Ts_ji_lookup = cp.asarray(Ts_ji_lookup)
            vk = cp.zeros(dms.shape)

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+3, dtype=np.int64)

            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            err = libpbc.PBC_build_k_init(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError(f'PBC build_k kernel init failed on Device {device_id}')
            kern = libpbc.PBC_build_k
            rys_envs = self.rys_envs
            rsjk_omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j][:3]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l][3:]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(dm_counts), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_sup_bas_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff), ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(cell.nbas),
                    supmol._bas.ctypes, ctypes.c_double(rsjk_omega))
                if err != 0:
                    raise RuntimeError(f'PBC_build_k kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()

            if kpts_band is not None:
                raise NotImplementedError

            if hermi == 0:
                n = dm_counts // 2
                vk[:n] += vk[n:,img_conj_mapping].transpose(0,1,3,2)
                vk = vk[:n]
            return vk, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        vk = multi_gpu.array_reduce([x[0] for x in results], inplace=True)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            _TimingCollector.summary(log.debug1, (x[2] for x in results))

        if not is_gamma_point:
            expLkz = expLk.view(np.float64).reshape(-1, nkpts, 2)
            vk = contract('sLmn,Lkz->skmnz', vk, expLkz)
            vk = cp.asarray(vk, order='C').view(np.complex128)[:,:,:,:,0]

        if not is_real:
            vk = vk[:n_dm] + vk[n_dm:] * 1j

        vk = vk.reshape(-1,nao,nao)
        if hermi == 1:
            vk = transpose_sum(vk)
        vk = cell.apply_CT_mat_C(vk)

        # When the vk_sr is evaluated in real space, the G=0 component is
        # included in vk_sr. This G=0 contribution will be handled in the vk_lr.
        # However, vk_lr may be skipped for certain RSH funcitonals like HSE06.
        # In this particular case (self.omega == omega and lr_factor == 0),
        # explicitly handle the G=0 term here.
        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if ((self.omega == omega and lr_factor == 0 and not exclude_dd_block) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            assert len(member(np.zeros(3), kpts)) > 0
            # difference associated to the G=0 term between the real space
            # integrals and the AFT integrals
            vk = vk.reshape(n_dm, nkpts, nao_orig, nao_orig)
            dms = dm.reshape(n_dm, nkpts, nao_orig, nao_orig)
            # Remove the G=0 contribution to match the output of FFTDF.get_jk().
            wcoulG_SR_at_G0 = -np.pi / omega**2 / cell.vol
            if exxdiv == 'ewald':
                # probe_charge_sr_coulomb equals to -2*ewovrl.
                # This term rapidly decays to 0 for large k-mesh. In the
                # FFTDF.get_jk based implementation, this contribution is
                # included in the short-range part.
                wcoulG_SR_at_G0 += nkpts*pbctools.madelung(cell, kpts, omega=-omega)

            s = int1e.int1e_ovlp(cell, kpts)
            for i in range(n_dm):
                for k in range(nkpts):
                    vk[i,k] += s[k].dot(dms[i,k]).dot(s[k]) * wcoulG_SR_at_G0

        if not is_gamma_point:
            weight = 1. / nkpts
            vk *= weight
        if sr_factor is not None and sr_factor != 1:
            vk *= sr_factor

        if kpts_band is None:
            vk = vk.reshape(dm.shape)
        else:
            raise NotImplementedError
        return vk

    def _get_k_lr(self, dm, hermi, kpts=None, kpts_band=None, exxdiv=None,
                  omega=None, lr_factor=1, sr_factor=1):
        if kpts_band is not None:
            raise NotImplementedError

        log = logger.new_logger(self)
        cpu0 = cpu1 = log.init_timer()
        cell = self.cell
        assert cell.dimension == 3

        omega, lr_factor, sr_factor = _check_rsh_factors(cell.cell, omega, lr_factor, sr_factor)
        omega = abs(omega)

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=True)
        log.debug('bvk_kmesh = %s', kmesh)
        bvk_ncells = np.prod(kmesh)

        mo_coeff = getattr(dm, 'mo_coeff', None)
        mo_occ = getattr(dm, 'mo_occ', None)
        dm = cp.asarray(dm)
        dms = _format_dms(dm, kpts)
        n_dm, nkpts, nao = dms.shape[:3]

        vk = cp.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            raise NotImplementedError

        if bvk_ncells == nkpts:
            kpt_iters = ((kpts[kp], ki_idx, kj_idx, kp==kp_conj)
                         for kp, kp_conj, ki_idx, kj_idx in bvk_kk_adapted_iter(kmesh))
            t_rev_pairs = conj_images_in_bvk_cell(kmesh, return_pair=True)
        else:
            raise NotImplementedError
        log.debug1('Num time-reversal pairs %d', len(t_rev_pairs))

        time_reversal_symmetry = self.time_reversal_symmetry
        if time_reversal_symmetry:
            for k, k_conj in t_rev_pairs:
                if (k != k_conj and abs(dms[:,k_conj] - dms[:,k].conj()).max() > cell.precision*5e2):
                    time_reversal_symmetry = False
                    log.debug2('Disable time_reversal_symmetry')
                    break

        if time_reversal_symmetry:
            k_to_compute = np.zeros(nkpts, dtype=np.int8)
            k_to_compute[t_rev_pairs[:,0]] = 1
        else:
            k_to_compute = np.ones(nkpts, dtype=np.int8)

        if mo_coeff is None:
            #dms = cell.apply_C_mat_CT(dms.reshape(-1,nao,nao))
            #dms = dms.reshape(n_dm, nkpts, nao1, nao1)
            if dms.dtype != vk.dtype:
                dms = dms.astype(vk.dtype)
            update_vk = aft_jk._update_vk_

            nao1 = cell.nao
            Gpq_unit = nao**2*bvk_ncells
            unit = (nao1**2*bvk_ncells + # Gpq
                    max(nao1**2*bvk_ncells,
                        (Gpq_unit + # Gpq_conj
                         Gpq_unit + # Gpq_conj[kj_idx]
                         n_dm*nkpts*nao1**2))) # contract('ngij,snjk->sngik', Gpq, dms)
        else:
            # dm ~= dm_factor * dm_factor.T
            # mo_coeff, mo_occ may not be a list of aligned array if
            # remove_lin_dep was applied to scf object.
            # We assume they are of the same length in this version.
            mo_coeff = cp.asarray(mo_coeff)
            mo_occ = cp.asarray(mo_occ)
            if is_single_kpt:
                if mo_coeff.ndim == 3:
                    mo_coeff = mo_coeff[:,None]
                    mo_occ = mo_occ[:,None]
                else:
                    mo_coeff = mo_coeff[None]
                    mo_occ = mo_occ[None]
            nocc = int((mo_occ > 0).sum(axis=-1).max().get())
            if mo_coeff.ndim == 4:  # KUHF
                occs = cp.array(mo_occ[:,:,:nocc], dtype=np.float64)
                dm_factor = cp.array(mo_coeff[:,:,:,:nocc],
                                     dtype=np.complex128, order='C', copy=True)
            else:  # KRHF
                occs = cp.asarray(mo_occ[None,:,:nocc], dtype=np.float64)
                dm_factor = cp.array(mo_coeff[None,:,:,:nocc],
                                     dtype=np.complex128, order='C', copy=True)
            dm_factor *= cp.sqrt(occs)[:,:,None,:]
            dms, dm_factor = dm_factor, None

            nao1 = cell.nao
            unit = (nao1**2*bvk_ncells + # Gpq
                    max(nao1**2*bvk_ncells,
                        (nao**2*bvk_ncells + # Gpq_conj
                         n_dm*nkpts*nao1*nocc*2))) # contract('ngij,snjk->sngik', Gpq, dms)

            log.debug2('time_reversal_symmetry = %s bvk_ncells = %d '
                       'cell0_nao = %d nocc = %d n_dm = %d',
                       time_reversal_symmetry, bvk_ncells, nao, nocc, n_dm)
            update_vk = aft_jk._update_vk_dmf
        log.debug2('set update_vk to %s', update_vk)

        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if exclude_dd_block:
            diffuse_i, diffuse_j = divmod(self.dd_ao_idx, nao1)
            unit += nao**2*bvk_ncells

        ft_opt = FTOpt(cell, kmesh)
        # permutation_symmetry between bra-in-cell0 and ket-in-bvkcell currently
        # only supports the complete set of kpts within MP mesh.
        ft_opt.permutation_symmetry = bvk_ncells == nkpts
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=False, kpts=kpts)

        mesh = self.mesh
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        # vk can be scale by the Nk weight, by including in coulG weights
        kws /= nkpts

        avail_mem = int(get_avail_mem(exclude_memory_pool=True) * .9)
        avail_mem -= n_dm*nkpts*nao1**2 * 16 # intermediates for vk or dms
        Gblksize = max(16, int(avail_mem/(16*unit))//8*8)
        Gblksize = min(Gblksize, ngrids, 16384)
        log.debug1('Gblksize = %d', Gblksize)

        Gpq_buf = cp.empty(unit*Gblksize + n_dm*nkpts*nao1**2, dtype=np.complex128)
        buf = Gpq_buf[nao1**2*bvk_ncells*Gblksize:]
        if exclude_dd_block:
            Gpq1_buf, buf = buf, buf[nao**2*bvk_ncells*Gblksize:]
        else:
            Gpq1_buf = Gpq_buf
        for group_id, (kpt, ki_idx, kj_idx, self_conj) in enumerate(kpt_iters):
            wcoulG, wcoulG_SR = _get_vk_wcoulG_and_SR(
                cell, kpt, kpts, exxdiv, mesh, Gv, kws, self.omega, omega, lr_factor, sr_factor)
            wcoulG_SR *= -1
            if not exclude_dd_block:
                wcoulG += wcoulG_SR

            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                log.debug3('update_vk [%s:%s]', p0, p1)
                Gpq = ft_kern(Gv[p0:p1], kpt, kj_idx=kj_idx, out=Gpq_buf, buf=buf)
                Gpq1 = _bas_recontract_ft_pair(cell, Gpq, Gpq1_buf, buf)
                update_vk(vk, Gpq1, dms, wcoulG[p0:p1], ki_idx, kj_idx,
                          not self_conj, k_to_compute, t_rev_pairs, buf)
                if exclude_dd_block:
                    Gpq[:,:,diffuse_i,diffuse_j] = 0.
                    Gpq1 = _bas_recontract_ft_pair(cell, Gpq, Gpq1_buf, buf)
                    update_vk(vk, Gpq1, dms, wcoulG_SR[p0:p1], ki_idx, kj_idx,
                              not self_conj, k_to_compute, t_rev_pairs, buf)
                Gpq = Gpq1 = None
            cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)

        if is_zero(kpts) and not np.iscomplexobj(dm):
            vk = vk.real

        if time_reversal_symmetry:
            for k, k_conj in t_rev_pairs:
                if k != k_conj:
                    vk[:,k_conj] = vk[:,k].conj()
        log.timer_debug1('get_k_kpts', *cpu0)
        return vk.reshape(dm.shape)

    def weighted_coulG(self, kpt=None, exx=None, mesh=None, omega=None,
                       kpts=None, lr_factor=1, sr_factor=1):
        '''weighted LR Coulomb kernel. Mimic AFTDF.weighted_coulG'''
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if omega is None:
            omega = 0
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        wcoulG, wcoulG_SR = _get_vk_wcoulG_and_SR(
            cell, kpt, kpts, exx, mesh, Gv, kws, self.omega, omega, lr_factor, sr_factor)
        wcoulG -= wcoulG_SR
        return wcoulG

    def _get_j_sr(self, dm, hermi, kpts=None, kpts_band=None):
        '''
        Build kpts adapted K matrices
        Return a (*, nkpts, nao, nao) array.

        If the "kpts" is supplied as None or [[0,0,0]] (the gamma point), the K
        matrix is still evaluated as the k-point sampling case. The "nkpts"
        dimension is set to 1
        '''
        log = logger.new_logger(self)
        cell = self.cell
        assert cell.dimension == 3
        nao = cell.nao
        supmol = self.supmol
        assert hermi == 1

        dm = asarray(dm)
        nao_orig = dm.shape[-1]
        dms = cell.apply_C_mat_CT(dm.reshape(-1,nao_orig,nao_orig))

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=True)
        is_gamma_point = is_zero(kpts)
        is_real = True
        if is_gamma_point:
            if is_single_kpt:
                assert dms.dtype == np.float64
            else:
                dms = dms.real
            nkpts = 1
            ao_loc = asarray(cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            # Additional dimension for kpts
            dms = dms[:,None,:,:]
            n_dm = len(dms)
            nimgs = nimgs_uniq_pair = 1
            Ts_ji_lookup = cp.zeros((nimgs, nimgs))
            sup_bas_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.int32) % cell.nbas
        else:
            bvk_ncells = np.prod(kmesh)
            nimgs = len(supmol.Ls)
            # When the size of BvK cell is smaller than the supmol, it's more
            # efficient to represent dms/vk in BvK cell
            if bvk_ncells < 7*nimgs:
                sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_bvk(supmol, kmesh, kpts)
                nimgs = bvk_ncells
            else:
                sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_supermol(supmol, kpts)
            nimgs_uniq_pair, nkpts = expLk.shape
            dms = dms.reshape(-1, nkpts, nao, nao)
            n_dm = len(dms)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            if absmax(dms.imag) < cell.precision*5e2:
                dms = dms.real
                dms = cp.asarray(dms, order='C')
            else:
                is_real = False
                dms = cp.vstack([dms.real, dms.imag])
            dm_cond = _dm_cond_from_compressed_dm(supmol, dms)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        # more errors are potentially accumulated in J matrix
        log_cutoff = math.log(self.estimate_cutoff_with_penalty(cell.precision))
        dm_penalty = float(dm_cond.max())
        log.debug1('dm_penalty = %f', dm_penalty)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(supmol, 'diffuse')

        libpbc.PBC_build_j.restype = ctypes.c_int

        uniq_l_ctr = cell.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(cell.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(self)
            t1 = log.init_timer()
            dms = cp.asarray(dms)
            dm_cond = cp.asarray(dm_cond)
            dm_counts = len(dms)

            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            _sup_bas_idx = cp.asarray(sup_bas_idx)
            _Ts_ji_lookup = cp.asarray(Ts_ji_lookup)
            vj = cp.zeros(dms.shape)

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+3, dtype=np.int64)

            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            err = libpbc.PBC_build_j_init(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError(f'PBC build_j kernel init failed on Device {device_id}')
            kern = libpbc.PBC_build_j
            rys_envs = self.rys_envs
            rsjk_omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j][:3]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l][3:]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(vj.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(dm_counts), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_sup_bas_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff), ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(cell.nbas),
                    supmol._bas.ctypes, ctypes.c_double(rsjk_omega))
                if err != 0:
                    raise RuntimeError(f'PBC_build_j kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()

            if kpts_band is not None:
                raise NotImplementedError
            return vj, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        vj = multi_gpu.array_reduce([x[0] for x in results], inplace=True)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            _TimingCollector.summary(log.debug1, (x[2] for x in results))

        if not is_gamma_point:
            expLkz = expLk.view(np.float64).reshape(-1, nkpts, 2)
            vj = contract('sLmn,Lkz->skmnz', vj, expLkz)
            vj = cp.asarray(vj, order='C').view(np.complex128)[:,:,:,:,0]

        if not is_real:
            vj = vj[:n_dm] + vj[n_dm:] * 1j

        vj = vj.reshape(-1,nao,nao)
        vj *= 2
        vj = transpose_sum(vj)
        vj = cell.apply_CT_mat_C(vj)

        if not is_gamma_point:
            weight = 1. / nkpts
            vj *= weight

        if kpts_band is None:
            vj = vj.reshape(dm.shape)
        else:
            raise NotImplementedError
        return vj

    def _get_j_lr(self, dm, hermi, kpts=None, kpts_band=None):
        if kpts_band is not None:
            raise NotImplementedError

        log = logger.new_logger(self)
        cell = self.cell
        assert cell.dimension == 3

        if kpts is None:
            kpts = np.zeros((1,3))
            kmesh = [1] * 3
        else:
            kpts = kpts.reshape(-1, 3)
            kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)
        log.debug('bvk_kmesh = %s', kmesh)
        bvk_ncells = np.prod(kmesh)

        dm = cp.asarray(dm, order='C')
        dms = _format_dms(dm, kpts)
        n_dm, nkpts, nao = dms.shape[:3]
        dms = cell.apply_C_mat_CT(dms.reshape(-1,nao,nao))
        nao1 = dms.shape[-1]
        dms = dms.reshape(n_dm, nkpts, nao1, nao1)
        vj = cp.zeros((n_dm, nkpts, nao1, nao1), dtype=np.complex128)

        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if exclude_dd_block:
            diffuse_i, diffuse_j = divmod(self.dd_ao_idx, nao1)

        mesh = self.mesh
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        kws /= nkpts

        kpt_allow = np.zeros(3)
        wcoulG = get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv, wrap_around=True, omega=0)
        wcoulG *= kws
        wcoulG_SR = get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                              wrap_around=True, omega=-self.omega)
        wcoulG_SR[0] += np.pi / self.omega**2
        wcoulG_SR *= -kws
        if not exclude_dd_block:
            wcoulG += wcoulG_SR

        ft_opt = FTOpt(cell, kmesh).build()
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=False)

        avail_mem = int(get_avail_mem(exclude_memory_pool=True) * .8)
        blksize = int(avail_mem/(nao**2*bvk_ncells*16*2)) // 32 * 32
        if blksize == 0:
            raise RuntimeError('Insufficient GPU memory')
        blksize = min(blksize, ngrids)

        for p0, p1 in lib.prange(0, ngrids, blksize):
            Gpq = ft_kern(Gv[p0:p1], kpt_allow, kpts)
            aft_jk._update_vj_(vj, Gpq, dms, wcoulG[p0:p1])
            if exclude_dd_block:
                Gpq[:,:,diffuse_i,diffuse_j] = 0.
                aft_jk._update_vj_(vj, Gpq, dms, wcoulG_SR[p0:p1])

        if is_zero(kpts):
            vj = vj.real
        vj = cell.apply_CT_mat_C(vj.reshape(-1,nao1,nao1))
        return vj.reshape(dm.shape)

    def get_j(self, dm, hermi=0, kpts=None, kpts_band=None):
        '''Compute J matrix
        '''
        if self.supmol is None:
            self.build(kpts)
        vj = self._get_j_sr(dm, hermi, kpts, kpts_band)
        vj += self._get_j_lr(dm, hermi, kpts, kpts_band)
        return vj

    def _get_ejk_sr_ip1(self, dm, kpts=None, exxdiv=None, omega=None,
                        j_factor=1, lr_factor=1, sr_factor=1, verbose=None):
        '''Compute the derivatives of the short-range part of the aggregated
        J/K contribution. The aggregated J/K contribution is given by
        j_factor - k_factor / 2, where k_factor = sr_factor
        '''
        log = logger.new_logger(self, verbose)
        cell = self.cell
        assert cell.dimension == 3
        nao = cell.nao
        supmol = self.supmol

        dm = asarray(dm)
        nao_orig = dm.shape[-1]
        dms = cell.apply_C_mat_CT(dm.reshape(-1,nao_orig,nao_orig))
        # Symmetrize density matrices because 8-fold symmetry is utilized when
        # computing integrals. Fold the contribution of the upper triangular
        # part of the density matrices into the lower triangular part.
        dms = transpose_sum(dms)
        dms *= .5

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=True)
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
            # Add the dimension for kpts
            dms = dms[:,None,:,:]
            nimgs = nimgs_uniq_pair = 1
            Ts_ji_lookup = cp.zeros((nimgs, nimgs))
            sup_bas_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.int32) % cell.nbas
        else:
            bvk_ncells = np.prod(kmesh)
            nimgs = len(supmol.Ls)
            # When the size of BvK cell is smaller than the supmol, it's more
            # efficient to represent dms/vk in BvK cell
            if bvk_ncells < 7*nimgs:
                sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_bvk(supmol, kmesh, kpts)
                nimgs = bvk_ncells
            else:
                sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_supermol(supmol, kpts)
            nimgs_uniq_pair, nkpts = expLk.shape
            dms = dms.reshape(-1, nkpts, nao, nao)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            assert absmax(dms.imag) < cell.precision*5e2
            expLk = None
            dms = dms.real
            dms = cp.asarray(dms, order='C')
            dm_cond = _dm_cond_from_compressed_dm(supmol, dms)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        n_dm = len(dms)
        assert n_dm <= 2
        cutoff = self.estimate_cutoff_with_penalty(cell.precision**.5*1e-2)
        log_cutoff = math.log(cutoff)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(supmol, 'diffuse')

        omega, lr_factor, sr_factor = _check_rsh_factors(cell.cell, omega, lr_factor, sr_factor)
        omega = abs(omega)

        uniq_l_ctr = cell.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(cell.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(cell, verbose)
            t0 = log.init_timer()
            dms = cp.asarray(dms)
            dm_cond = cp.asarray(dm_cond)

            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            _sup_bas_idx = cp.asarray(sup_bas_idx)
            _Ts_ji_lookup = cp.asarray(Ts_ji_lookup)
            ejk = cp.zeros((cell.natm, 3))

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int64)
            dd_pool = cp.empty((workers, DD_CACHE_MAX), dtype=np.float64)

            t1 = log.timer_debug1(f'ejk_sr initialization on Device {device_id}', *t0)
            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            err = libpbc.PBC_build_jk_ip1_init(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError(f'PBC build_jk_ip1 kernel init failed on Device {device_id}')
            kern = libpbc.PBC_per_atom_jk_ip1
            rys_envs = self.rys_envs
            omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j][:3]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l][3:]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = _ejk_quartets_scheme(supmol, uniq_l_ctr[[i, j, k, l]])
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(sr_factor),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_sup_bas_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(cell.nbas),
                    supmol._bas.ctypes, ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError(f'PBC_build_jk_ip1 kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()
            return ejk, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            _TimingCollector.summary(log.debug1, (x[2] for x in results))

        ejk = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        ejk = ejk.get()

        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if ((self.omega == omega and j_factor == 0 and lr_factor == 0 and
             not exclude_dd_block) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
            # difference associated to the G=0 term between the real space
            # integrals and the AFT integrals
            dms = dm.reshape(n_dm, nkpts, nao_orig, nao_orig)
            omega = self.omega
            wcoulG_for_k = -np.pi / omega**2 / cell.vol
            if exxdiv == 'ewald':
                wcoulG_for_k += nkpts*pbctools.madelung(cell, kpts, omega=-omega)
            s0 = int1e.int1e_ovlp(cell, kpts)
            s1 = int1e.int1e_ipovlp(cell, kpts)
            k_dm = contract('nkpq,kqr->nkpr', dms, s0)
            k_dm = contract('nkpr,nkrs->kps', k_dm, dms)
            if n_dm == 1: # RHF
                k_dm *= .5 * sr_factor * wcoulG_for_k
            else:
                k_dm *= sr_factor * wcoulG_for_k
            ejk += contract_h1e_dm(cell.cell, s1, k_dm, hermi=1) * .5

        if not is_gamma_point:
            ejk *= 1. / nkpts**2
        return ejk

    def _get_ejk_lr_ip1(self, dm, kpts=None, omega=None, exxdiv=None,
                        j_factor=1, lr_factor=1, sr_factor=1):
        '''Compute the derivatives of the long-range part of the aggregated
        J/K contribution. The aggregated J/K contribution is given by
        j_factor*J-k_factor*K/2 for RHF and j_factor*J-k_factor*K for UHF.
        '''
        log = logger.new_logger(self)
        cell = self.cell
        assert cell.dimension == 3

        omega, lr_factor, sr_factor = _check_rsh_factors(cell.cell, omega, lr_factor, sr_factor)
        omega = abs(omega)

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=True)
        log.debug('bvk_kmesh = %s', kmesh)
        bvk_ncells = np.prod(kmesh)
        is_gamma_point = is_zero(kpts)

        dms = _format_dms(dm, kpts)
        n_dm, nkpts, nao = dms.shape[:3]
        assert nkpts == len(kpts)

        dms = cell.apply_C_mat_CT(dms.reshape(-1,nao,nao))
        nao = dms.shape[-1]
        dms = dms.reshape(n_dm,nkpts,nao,nao)

        if n_dm == 1: # RHF or KRHF
            # RHF energy is computed as J - 1/2 K
            lr_factor *= .5
            sr_factor *= .5
        elif n_dm > 2:
            raise NotImplementedError

        ft_opt = FTOpt(cell, kmesh)
        ft_opt.permutation_symmetry = bvk_ncells == nkpts
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=False, kpts=kpts)

        if not is_gamma_point:
            expLk = cp.exp(1j*cp.asarray(ft_opt.bvkmesh_Ls).dot(cp.asarray(kpts).T))

        mesh = self.mesh
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)

        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        shm_size = aft_jk._estimate_max_shm_size(cell, (1,0))
        log.debug('bas_ij_idx=%d nbatches=%d shm_size=%d',
                  len(bas_ij_idx), nbatches_shl_pair, shm_size)

        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if exclude_dd_block:
            bas_ij_wo_dd, img_idx_wo_dd, shl_pair_offsets_wo_dd = \
                    _generate_shl_pairs(ft_opt, self.dd_bas_idx)

        def get_j_ip1():
            t0 = log.init_timer()
            if n_dm == 1:
                dm_sf = dms[0]
            else:
                dm_sf = dms[0] + dms[1]
            if is_gamma_point:
                dms_bvkcell = cp.asarray(dm_sf.real, order='C')
            else:
                dms_bvkcell = contract('Lk,kpq->Lpq', expLk, dm_sf)
                assert abs(dms_bvkcell.imag).max() < 1e-6
                dms_bvkcell = cp.asarray(dms_bvkcell.real, order='C')

            # memory buffer required by eval_ft
            avail_mem = get_avail_mem(exclude_memory_pool=True) * .8
            blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
            blksize = min(blksize, ngrids, 16384)
            log.debug1('blksize=%d', blksize)

            if exclude_dd_block:
                diffuse_i, diffuse_j = divmod(self.dd_ao_idx, nao)

            kpt_allow = np.zeros(3)
            wcoulG = get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv, wrap_around=True, omega=0)
            wcoulG *= kws
            wcoulG_SR = get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                                  wrap_around=True, omega=-self.omega)
            wcoulG_SR[0] += np.pi / self.omega**2
            wcoulG_SR *= -kws
            if not exclude_dd_block:
                wcoulG += wcoulG_SR

            aft_envs = ft_opt.aft_envs
            kern = libpbc.PBC_ft_aopair_ej_ip1
            ej = cp.zeros((cell.natm, 3))
            for p0, p1 in lib.prange(0, ngrids, blksize):
                nGv = p1 - p0
                Gpq = ft_kern(Gv[p0:p1])
                Gpq = Gpq.transpose(0,2,3,1)
                vG = contract('kji,kijg->g', dm_sf, Gpq).conj()
                vG *= wcoulG[p0:p1]
                GvT = cp.asarray(Gv[p0:p1].T.ravel())
                err = kern(
                    ctypes.cast(ej.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms_bvkcell.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vG.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    ctypes.byref(aft_envs),
                    ctypes.c_int(nbatches_shl_pair),
                    ctypes.c_int(nGv),
                    ctypes.c_int(shm_size),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(int(ft_opt.permutation_symmetry)))
                if err != 0:
                    raise RuntimeError('PBC_ft_aopair_ej_ip1 failed')
                if exclude_dd_block and len(bas_ij_wo_dd) > 0:
                    Gpq[:,diffuse_i,diffuse_j] = 0.
                    vG = contract('kji,kijg->g', dm_sf, Gpq).conj()
                    vG *= wcoulG_SR[p0:p1]
                    err = kern(
                        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dms_bvkcell.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vG.data.ptr, ctypes.c_void_p),
                        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                        ctypes.byref(aft_envs),
                        ctypes.c_int(len(shl_pair_offsets_wo_dd) - 1),
                        ctypes.c_int(nGv),
                        ctypes.c_int(shm_size),
                        ctypes.cast(bas_ij_wo_dd.data.ptr, ctypes.c_void_p),
                        ctypes.cast(img_idx_wo_dd.data.ptr, ctypes.c_void_p),
                        ctypes.cast(shl_pair_offsets_wo_dd.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(int(ft_opt.permutation_symmetry)))
                    if err != 0:
                        raise RuntimeError('PBC_ft_aopair_ej_ip1 failed')
                Gpq = None
            if not ft_opt.permutation_symmetry:
                ej *= .5
            ej *= j_factor / nkpts**2
            ej = ej.get()
            log.timer_debug1('get_ej_ip1', *t0)
            return ej

        def get_k_ip1():
            cpu0 = cpu1 = log.init_timer()
            avail_mem = get_avail_mem(exclude_memory_pool=True) * .8
            blksize = int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16
            if blksize == 0:
                raise RuntimeError('Insufficient GPU memory')
            blksize = min(blksize, ngrids, 16384)
            log.debug1('blksize=%d', blksize)

            if exclude_dd_block:
                diffuse_i, diffuse_j = divmod(self.dd_ao_idx, nao)

            aft_envs = ft_opt.aft_envs
            kern = libpbc.PBC_ft_aopair_ek_ip1
            ek = cp.zeros((cell.natm, 3))
            for group_id, (kp, kp_conj, ki_idx, kj_idx) in enumerate(bvk_kk_adapted_iter(kmesh)):
                kpt = kpts[kp]
                wcoulG, wcoulG_SR = _get_vk_wcoulG_and_SR(
                    cell, kpt, kpts, exxdiv, mesh, Gv, kws, self.omega, omega, lr_factor, sr_factor)
                wcoulG_SR *= -1
                if not exclude_dd_block:
                    wcoulG += wcoulG_SR

                swap_2e = kp != kp_conj
                for p0, p1 in lib.prange(0, ngrids, blksize):
                    nGv = p1 - p0
                    Gpq = ft_kern(-Gv[p0:p1], -kpt, -kpts, kj_idx)
                    pqG_conj = Gpq.transpose(0,2,3,1)
                    if is_gamma_point:
                        tmp = contract('sjk,lkg->sjlg', dms[:,0], pqG_conj[0])
                        dm_vG = contract('sjlg,sli->jig', tmp, dms[:,0])
                        if ft_opt.permutation_symmetry:
                            dm_vG *= 2
                    else:
                        idx = np.empty_like(ki_idx)
                        idx[kj_idx] = ki_idx
                        tmp = contract('snjk,nlkg->snljg', dms, pqG_conj)
                        tmp = contract('snljg,snli->nijg', tmp, dms[:,idx])
                        dm_vG = contract('Lk,kijg->Ljig', expLk, tmp)
                        if ft_opt.permutation_symmetry:
                            dm_vG += contract('Lk,kijg->Lijg', expLk[:,idx].conj(), tmp)
                    if swap_2e:
                        dm_vG *= wcoulG[p0:p1] * 2
                    else:
                        dm_vG *= wcoulG[p0:p1]
                    dm_vG = cp.asarray(dm_vG, order='C')
                    GvT = cp.asarray((Gv[p0:p1]+kpt).T.ravel())
                    err = kern(
                        ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                        ctypes.byref(aft_envs),
                        ctypes.c_int(nbatches_shl_pair),
                        ctypes.c_int(nGv),
                        ctypes.c_int(shm_size),
                        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(int(ft_opt.permutation_symmetry)))
                    if err != 0:
                        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')

                    if exclude_dd_block and len(bas_ij_wo_dd) > 0:
                        pqG_conj[:,diffuse_i,diffuse_j] = 0.
                        if is_gamma_point:
                            tmp = contract('sjk,lkg->sjlg', dms[:,0], pqG_conj[0])
                            dm_vG = contract('sjlg,sli->jig', tmp, dms[:,0])
                            if ft_opt.permutation_symmetry:
                                dm_vG *= 2
                        else:
                            tmp = contract('snjk,nlkg->snljg', dms, pqG_conj)
                            tmp = contract('snljg,snli->nijg', tmp, dms[:,idx])
                            dm_vG = contract('Lk,kijg->Ljig', expLk, tmp)
                            if ft_opt.permutation_symmetry:
                                dm_vG += contract('Lk,kijg->Lijg', expLk[:,idx].conj(), tmp)
                        if swap_2e:
                            dm_vG *= wcoulG_SR[p0:p1] * 2
                        else:
                            dm_vG *= wcoulG_SR[p0:p1]
                        dm_vG = cp.asarray(dm_vG, order='C')
                        err = kern(
                            ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                            ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                            ctypes.byref(aft_envs),
                            ctypes.c_int(len(shl_pair_offsets_wo_dd) - 1),
                            ctypes.c_int(nGv),
                            ctypes.c_int(shm_size),
                            ctypes.cast(bas_ij_wo_dd.data.ptr, ctypes.c_void_p),
                            ctypes.cast(img_idx_wo_dd.data.ptr, ctypes.c_void_p),
                            ctypes.cast(shl_pair_offsets_wo_dd.data.ptr, ctypes.c_void_p),
                            ctypes.c_int(int(ft_opt.permutation_symmetry)))
                        if err != 0:
                            raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
                    Gpq = pqG_conj = tmp = dm_vG = None
                cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)
            ek *= .5 / nkpts**2
            ek = ek.get()
            log.timer_debug1('get_ek_ip1', *cpu0)
            return ek

        ej = ek = 0
        if j_factor != 0:
            ej = get_j_ip1()
        if lr_factor != 0 or sr_factor != 0:
            ek = get_k_ip1()
        return ej - ek

    def _get_ejk_sr_strain_deriv(self, dm, kpts=None, exxdiv=None, omega=None,
                        j_factor=1, lr_factor=1, sr_factor=1, verbose=None):
        '''Compute the derivatives of the short-range part of the aggregated
        J/K contribution. The aggregated J/K contribution is given by
        j_factor - k_factor / 2.
        '''
        log = logger.new_logger(self, verbose)
        cell = self.cell
        assert cell.dimension == 3
        nao = cell.nao
        supmol = self.supmol

        dm = asarray(dm)
        nao_orig = dm.shape[-1]
        dms = cell.apply_C_mat_CT(dm.reshape(-1,nao_orig,nao_orig))
        # Symmetrize density matrices because 8-fold symmetry is utilized when
        # computing integrals. Fold the contribution of the upper triangular
        # part of the density matrices into the lower triangular part.
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
            # Add the dimension for kpts
            dms = dms[:,None,:,:]
            nimgs = len(supmol.Ls)
            nimgs_uniq_pair = 1
            Ts_ji_lookup = cp.zeros((nimgs, nimgs))
            sup_bas_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.int32)
        else:
            sup_bas_idx, Ts_ji_lookup, expLk = _double_latsum_in_supermol(supmol, kpts)
            nimgs = len(supmol.Ls)
            nimgs_uniq_pair, nkpts = expLk.shape
            dms = dms.reshape(-1, nkpts, nao, nao)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            assert absmax(dms.imag) < cell.precision*5e2
            expLk = None
            dms = dms.real
            dms = cp.asarray(dms, order='C')
            dm_cond = _dm_cond_from_compressed_dm(supmol, dms)
        dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        n_dm = len(dms)
        assert n_dm <= 2
        cutoff = self.estimate_cutoff_with_penalty(cell.precision**.5*1e-2)
        log_cutoff = math.log(cutoff)

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(supmol, 'diffuse')

        omega, lr_factor, sr_factor = _check_rsh_factors(cell.cell, omega, lr_factor, sr_factor)
        omega = abs(omega)

        uniq_l_ctr = cell.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(cell.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        tasks = ((i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1))

        def proc(dms, dm_cond):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(cell, verbose)
            t0 = log.init_timer()
            dms = cp.asarray(dms)
            dm_cond = cp.asarray(dm_cond)

            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            _sup_bas_idx = cp.asarray(sup_bas_idx)
            _Ts_ji_lookup = cp.asarray(Ts_ji_lookup)
            ejk = cp.zeros((cell.natm, 3))
            sigma = cp.zeros((3, 3))

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int64)
            dd_pool = cp.empty((workers, DD_CACHE_MAX), dtype=np.float64)

            t1 = log.timer_debug1(f'ejk_sr_strain_deriv initialization on Device {device_id}', *t0)
            timing_collection = _TimingCollector(log.timer_debug1)
            kern_counts = 0
            err = libpbc.PBC_build_jk_ip1_init(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError(f'PBC build_jk_ip1 kernel init failed on Device {device_id}')
            kern = libpbc.PBC_jk_strain_deriv
            rys_envs = self.rys_envs
            omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j][:3]
                pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l][3:]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = _ejk_quartets_scheme(supmol, uniq_l_ctr[[i, j, k, l]])
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(sr_factor),
                    ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_sup_bas_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(cell.nbas),
                    supmol._bas.ctypes, ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError(f'PBC_jk_strain_deriv kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()
            return ejk, sigma, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        dms = None

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[2] for x in results))
            _TimingCollector.summary(log.debug1, (x[3] for x in results))

        sigma = multi_gpu.array_reduce([x[1] for x in results], inplace=True)
        sigma = sigma.get()
        sigma *= 2 / nkpts**2

        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if ((self.omega == omega and j_factor == 0 and lr_factor == 0 and
             not exclude_dd_block) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            raise
            from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
            # difference associated to the G=0 term between the real space
            # integrals and the AFT integrals
            dm0 = dm.reshape(n_dm, nkpts, nao_orig, nao_orig)
            omega = self.omega
            wcoulG_for_k = -np.pi / omega**2 / cell.vol
            if exxdiv == 'ewald':
                from gpu4pyscf.pbc.df.aft_jk import _exxdiv_ewald_strain_deriv
                exx_0, exx_1 = _exxdiv_ewald_strain_deriv(cell, kpts, -omega)
                wcoulG_for_k += exx_0
            s0 = int1e.int1e_ovlp(cell, kpts)
            k_dm = contract('nkpq,kqr->nkpr', dm0, s0)
            k_dm = contract('nkpr,nkrs->kps', k_dm, dm0)
            ek_G0 = .5 / nkpts**2 * cp.einsum('kij,kji->', s0, k_dm).real.get()
            if n_dm == 1: # RHF
                sr_factor *= .5
            k_dm *= sr_factor * wcoulG_for_k / nkpts
            ek_G0 *= sr_factor

            # Response of the overlap integrals in Tr(S D S D)
            int1e_opt = int1e._Int1eOpt(cell, 1)
            # *2 due to (d/dX ij|kl) + (ij|d/dX kl)
            # scaled by 1/nkpts only instead of 1/nkpts**2 because
            # get_ovlp_strain_deriv has already scaled the output by 1/nkpts
            sigma += 2 / nkpts * int1e_opt.get_ovlp_strain_deriv(k_dm, kpts)
            if exxdiv == 'ewald':
                exx_1 *= ek_G0
                sigma += exx_1
        return sigma

    def _get_ejk_lr_strain_deriv(self, dm, kpts=None, omega=None, exxdiv=None,
                        j_factor=1, lr_factor=1, sr_factor=1):
        '''Compute the strain derivatives of the long-range part of the
        aggregated J/K contribution. The aggregated J/K contribution is given by
        j_factor*J-k_factor*K/2 for RHF and j_factor*J-k_factor*K for UHF.
        '''
        from gpu4pyscf.pbc.grad.rks_stress import (
            _get_weighted_coulG_strain_derivatives as get_wcoulG)
        from gpu4pyscf.pbc.df.aft_jk import _exxdiv_ewald_strain_deriv
        log = logger.new_logger(self)
        cell = self.cell
        assert cell.dimension == 3

        omega, lr_factor, sr_factor = _check_rsh_factors(cell.cell, omega, lr_factor, sr_factor)
        omega = abs(omega)

        kpts, is_single_kpt = _check_kpts(kpts, dm)
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=True)
        log.debug('bvk_kmesh = %s', kmesh)
        bvk_ncells = np.prod(kmesh)
        is_gamma_point = is_zero(kpts)

        dm0 = _format_dms(dm, kpts)
        n_dm, nkpts, nao = dm0.shape[:3]
        assert nkpts == len(kpts)

        dms = cell.apply_C_mat_CT(dm0.reshape(-1,nao,nao))
        nao = dms.shape[-1]
        dms = dms.reshape(n_dm,nkpts,nao,nao)

        if n_dm == 1: # RHF or KRHF
            # RHF energy is computed as J - 1/2 K
            lr_factor *= .5
            sr_factor *= .5
        elif n_dm > 2:
            raise NotImplementedError

        ft_opt = FTOpt(cell, kmesh)
        ft_opt.permutation_symmetry = bvk_ncells == nkpts
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=False, kpts=kpts)

        if not is_gamma_point:
            expLk = cp.exp(1j*cp.asarray(ft_opt.bvkmesh_Ls).dot(cp.asarray(kpts).T))

        mesh = self.mesh
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)

        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        shm_size = aft_jk._estimate_max_shm_size(cell, (1,0))
        log.debug('bas_ij_idx=%d nbatches=%d shm_size=%d',
                  len(bas_ij_idx), nbatches_shl_pair, shm_size)

        exclude_dd_block = self.exclude_dd_block and len(self.dd_ao_idx) > 0
        if exclude_dd_block:
            bas_ij_wo_dd, img_idx_wo_dd, shl_pair_offsets_wo_dd = \
                    _generate_shl_pairs(ft_opt, self.dd_bas_idx)

        def get_j_sigma():
            t0 = log.init_timer()
            if n_dm == 1:
                dm_sf = dms[0]
            else:
                dm_sf = dms[0] + dms[1]
            if is_gamma_point:
                dms_bvkcell = cp.asarray(dm_sf.real, order='C')
            else:
                dms_bvkcell = contract('Lk,kpq->Lpq', expLk, dm_sf)
                assert abs(dms_bvkcell.imag).max() < 1e-6
                dms_bvkcell = cp.asarray(dms_bvkcell.real, order='C')

            # memory buffer required by eval_ft
            avail_mem = get_avail_mem(exclude_memory_pool=True) * .8
            blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
            blksize = min(blksize, ngrids, 16384)
            log.debug1('blksize=%d', blksize)

            if exclude_dd_block:
                diffuse_i, diffuse_j = divmod(self.dd_ao_idx, nao)

            wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gv, 0)
            wcoulG_SR_0, wcoulG_SR_1 = get_wcoulG(cell, Gv, -self.omega)
            wcoulG_SR_at_G0 = np.pi / self.omega**2 * kws
            wcoulG_SR_0[0] += wcoulG_SR_at_G0
            wcoulG_SR_1[:,:,0] -= wcoulG_SR_at_G0 * cp.eye(3)
            wcoulG_SR_0 *= -1
            wcoulG_SR_1 *= -1
            if not exclude_dd_block:
                wcoulG_0 += wcoulG_SR_0
                wcoulG_1 += wcoulG_SR_1

            aft_envs = ft_opt.aft_envs
            kern = libpbc.PBC_ft_aopair_ej_strain_deriv
            ej = cp.zeros((cell.natm, 3))
            sigma = cp.zeros((3, 3))
            for p0, p1 in lib.prange(0, ngrids, blksize):
                nGv = p1 - p0
                Gpq = ft_kern(Gv[p0:p1])
                Gpq = Gpq.transpose(0,2,3,1)
                rhoG = contract('kji,kijg->g', dm_sf, Gpq)
                sigma += .25*cp.einsum('xyg,g,g->xy', wcoulG_1[:,:,p0:p1], rhoG.conj(), rhoG).real
                vG = rhoG.conj()
                vG *= wcoulG_0[p0:p1]
                GvT = cp.asarray(Gv[p0:p1].T.ravel())
                err = kern(
                    ctypes.cast(ej.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms_bvkcell.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vG.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    ctypes.byref(aft_envs),
                    ctypes.c_int(nbatches_shl_pair),
                    ctypes.c_int(nGv),
                    ctypes.c_int(shm_size),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(int(ft_opt.permutation_symmetry)))
                if err != 0:
                    raise RuntimeError('PBC_ft_aopair_ej_strain_deriv failed')
                if exclude_dd_block and len(bas_ij_wo_dd) > 0:
                    Gpq[:,diffuse_i,diffuse_j] = 0.
                    rhoG = contract('kji,kijg->g', dm_sf, Gpq)
                    sigma += .25*cp.einsum('xyg,g,g->xy', wcoulG_SR_1[:,:,p0:p1], rhoG.conj(), rhoG).real
                    vG = rhoG.conj()
                    vG *= wcoulG_SR_0[p0:p1]
                    err = kern(
                        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
                        ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dms_bvkcell.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vG.data.ptr, ctypes.c_void_p),
                        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                        ctypes.byref(aft_envs),
                        ctypes.c_int(len(shl_pair_offsets_wo_dd) - 1),
                        ctypes.c_int(nGv),
                        ctypes.c_int(shm_size),
                        ctypes.cast(bas_ij_wo_dd.data.ptr, ctypes.c_void_p),
                        ctypes.cast(img_idx_wo_dd.data.ptr, ctypes.c_void_p),
                        ctypes.cast(shl_pair_offsets_wo_dd.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(int(ft_opt.permutation_symmetry)))
                    if err != 0:
                        raise RuntimeError('PBC_ft_aopair_ej_strain_deriv failed')
                Gpq = None
            sigma *= 2 * j_factor / nkpts**2
            sigma = sigma.get()
            log.timer_debug1('get_ej_strain_deriv', *t0)
            return sigma

        def get_k_sigma():
            cpu0 = cpu1 = log.init_timer()
            avail_mem = get_avail_mem(exclude_memory_pool=True) * .8
            blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
            if blksize == 0:
                raise RuntimeError('Insufficient GPU memory')
            blksize = min(blksize, ngrids, 16384)
            log.debug1('blksize=%d', blksize)

            if exclude_dd_block:
                diffuse_i, diffuse_j = divmod(self.dd_ao_idx, nao)

            aft_envs = ft_opt.aft_envs
            kern = libpbc.PBC_ft_aopair_ek_strain_deriv
            ek = cp.zeros((cell.natm, 3))
            sigma = cp.zeros((3, 3))
            sigma1 = cp.zeros((3, 3))
            for group_id, (kp, kp_conj, ki_idx, kj_idx) in enumerate(bvk_kk_adapted_iter(kmesh)):
                kpt = kpts[kp]
                Gvk = Gv + kpt
                remove_G0 = is_zero(kpt)
                wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gvk, 0)
                if remove_G0 and exxdiv == 'ewald':
                    fr_ewald_0, fr_ewald_1 = aft_jk._exxdiv_ewald_strain_deriv(cell, kpts, 0.)
                    wcoulG_0[0] += fr_ewald_0
                    wcoulG_1[:,:,0] += cp.asarray(fr_ewald_1)
                if lr_factor == sr_factor:
                    wcoulG_0 *= lr_factor
                    wcoulG_1 *= lr_factor
                else:
                    wcoulG_LR_0, wcoulG_LR_1 = get_wcoulG(cell, Gvk, omega)
                    if remove_G0 and exxdiv == 'ewald':
                        lr_ewald_0, lr_ewald_1 = aft_jk._exxdiv_ewald_strain_deriv(cell, kpts, omega)
                        wcoulG_LR_0[0] += lr_ewald_0
                        wcoulG_LR_1[:,:,0] += cp.asarray(lr_ewald_1)
                    wcoulG_0 -= wcoulG_LR_0
                    wcoulG_0 *= sr_factor
                    wcoulG_0 += wcoulG_LR_0 * lr_factor
                    wcoulG_1 -= wcoulG_LR_1
                    wcoulG_1 *= sr_factor
                    wcoulG_1 += wcoulG_LR_1 * lr_factor
                wcoulG_SR_0, wcoulG_SR_1 = get_wcoulG(cell, Gvk, -self.omega)
                if remove_G0:
                    wcoulG_SR_at_G0 = np.pi / self.omega**2 * kws
                    wcoulG_SR_0[0] += wcoulG_SR_at_G0
                    wcoulG_SR_1[:,:,0] -= wcoulG_SR_at_G0 * cp.eye(3)
                wcoulG_SR_0 *= -sr_factor
                wcoulG_SR_1 *= -sr_factor
                if not exclude_dd_block:
                    wcoulG_0 += wcoulG_SR_0
                    wcoulG_1 += wcoulG_SR_1

                swap_2e = kp != kp_conj
                for p0, p1 in lib.prange(0, ngrids, blksize):
                    nGv = p1 - p0
                    Gpq = ft_kern(Gv[p0:p1], kpt, kj_idx=kj_idx)
                    Gpq = Gpq.transpose(0,2,3,1)
                    Gpq_conj = Gpq.conj()
                    if is_gamma_point:
                        tmp = contract('sjk,lkg->sjlg', dms[:,0], Gpq_conj[0])
                        dm_vG = contract('sjlg,sli->jig', tmp, dms[:,0])
                        vkG = cp.einsum('pqg,qpg->g', dm_vG, Gpq[0]).real
                    else:
                        idx = np.empty_like(ki_idx)
                        idx[kj_idx] = ki_idx
                        dm_k = contract('snjk,nlkg->snjlg', dms, Gpq_conj)
                        dm_k = contract('snjlg,snli->njig', dm_k, dms[:,idx])
                        dm_vG = contract('Lk,kpqg->Lpqg', expLk, dm_k)
                        vkG = cp.einsum('njig,nijg->g', dm_k, Gpq).real
                    tmp = cp.einsum('xyg,g->xy', wcoulG_1[:,:,p0:p1], vkG)
                    if swap_2e:
                        sigma += tmp * 2
                        dm_vG *= wcoulG_0[p0:p1] * 2
                    else:
                        sigma += tmp
                        dm_vG *= wcoulG_0[p0:p1]
                    dm_vG = cp.asarray(dm_vG, order='C')
                    GvT = cp.asarray(Gvk[p0:p1].T.ravel())
                    err = kern(
                        ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                        ctypes.cast(sigma1.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                        ctypes.byref(aft_envs),
                        ctypes.c_int(nbatches_shl_pair),
                        ctypes.c_int(nGv),
                        ctypes.c_int(shm_size),
                        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(int(ft_opt.permutation_symmetry)))
                    if err != 0:
                        raise RuntimeError('PBC_ft_aopair_ek_strain_deriv failed')

                    if exclude_dd_block and len(bas_ij_wo_dd) > 0:
                        Gpq[:,diffuse_i,diffuse_j] = 0.
                        Gpq_conj[:,diffuse_i,diffuse_j] = 0.
                        if is_gamma_point:
                            tmp = contract('sjk,lkg->sjlg', dms[:,0], Gpq_conj[0])
                            dm_vG = contract('sjlg,sli->jig', tmp, dms[:,0])
                            vkG = cp.einsum('pqg,qpg->g', dm_vG, Gpq[0]).real
                        else:
                            idx = np.empty_like(ki_idx)
                            idx[kj_idx] = ki_idx
                            dm_k = contract('snjk,nlkg->snjlg', dms, Gpq_conj)
                            dm_k = contract('snjlg,snli->njig', dm_k, dms[:,idx])
                            dm_vG = contract('Lk,kpqg->Lpqg', expLk, dm_k)
                            vkG = cp.einsum('njig,nijg->g', dm_k, Gpq).real
                        tmp = cp.einsum('xyg,g->xy', wcoulG_SR_1[:,:,p0:p1], vkG)
                        if swap_2e:
                            sigma += tmp * 2
                            dm_vG *= wcoulG_SR_0[p0:p1] * 2
                        else:
                            sigma += tmp
                            dm_vG *= wcoulG_SR_0[p0:p1]
                        dm_vG = cp.asarray(dm_vG, order='C')
                        err = kern(
                            ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                            ctypes.cast(sigma1.data.ptr, ctypes.c_void_p),
                            ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                            ctypes.byref(aft_envs),
                            ctypes.c_int(len(shl_pair_offsets_wo_dd) - 1),
                            ctypes.c_int(nGv),
                            ctypes.c_int(shm_size),
                            ctypes.cast(bas_ij_wo_dd.data.ptr, ctypes.c_void_p),
                            ctypes.cast(img_idx_wo_dd.data.ptr, ctypes.c_void_p),
                            ctypes.cast(shl_pair_offsets_wo_dd.data.ptr, ctypes.c_void_p),
                            ctypes.c_int(int(ft_opt.permutation_symmetry)))
                        if err != 0:
                            raise RuntimeError('PBC_ft_aopair_ek_strain_deriv failed')
                    Gpq = Gpq_conj = dm_k = tmp = dm_vG = None
                cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)
            sigma *= 1. / nkpts**2
            # First *2 due to i>=j symmetry in kernel;
            # second *2 due to (d/dX ij|kl) + (ij|d/dX kl)
            sigma1 *= 2 * 2 / nkpts**2
            sigma += sigma1
            sigma = sigma.get()
            sigma *= .5 # *.5 for the factor 1/2 in Coulomb operator
            log.timer_debug1('get_ek_strain_deriv', *cpu0)
            return sigma

        ej = ek = 0
        if j_factor != 0:
            ej = get_j_sigma()
        if lr_factor != 0 or sr_factor != 0:
            ek = get_k_sigma()
        return ej - ek

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
        # image (-img_i + img_j) ~ Ts_lookup[img_i, img_j] == index of Ts
        self.double_latsum_Ts = None
        self.Ts_ji_lookup = None

    @classmethod
    def from_cell(cls, cell, omega, rcut=None, verbose=None):
        log = logger.new_logger(cell, verbose)
        if cell.dimension == 0:
            raise NotImplementedError

        if rcut is None:
            rcut = estimate_rcut(cell, omega)
        rcut_max = rcut.max()
        Ls = cell.get_lattice_Ls(rcut=rcut_max)
        Ls = Ls[np.linalg.norm(Ls-.1, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug1('Generate supmol. omega = %g rcut = %g nimgs = %d',
                   omega, rcut_max, nimgs)

        supmol = cls()
        supmol.__dict__.update(cell.__dict__)
        supmol = pbctools._build_supcell_(supmol, cell, Ls)
        supmol.cell = cell
        supmol.Ls = Ls
        supmol.precision = cell.precision
        supmol._env[gto.PTR_EXPCUTOFF] = -np.log(cell.precision*1e-8)
        supmol.omega = -abs(omega) # Use supmol to handle SR integrals only

        rcut_for_atoms = asarray(groupby(cell._bas[:,gto.ATOM_OF], rcut, 'max'))
        # Search the shortest distance to the reference cell for each atom in the supercell.
        atom_coords = supmol.atom_coords()
        atm_idx = np.unique(cell._bas[:,gto.ATOM_OF])
        d = dist_matrix(atom_coords, cell.atom_coords()[atm_idx])
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
        return supmol

def estimate_rcut(cell, omega, precision=None):
    '''Estimate rcut for 2e SR-integrals

    This function is generally based on the implementation of
    pyscf.pbc.scf.rsjk.estimate_rcut with small modifications in compact and
    diffuse bases partition.
    '''
    if precision is None:
        precision = cell.precision * 1e-2

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
    fac /= aij**(li+1.5) * akl**(lk+1.5) * aj**lj * al**ll
    fac *= fl / precision

    vol = cell.vol
    lat_unit = vol**(1./3)
    rad = cell.rcut / lat_unit + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*(cell.rcut)/(vol*theta) + surface
    fac *= lattice_sum_factor

    r0 = cell.rcut
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
    rcut = r0
    return rcut

def _search_diffuse_pairs(cell, mesh):
    '''Return a mask identifying orbital pairs that can be converged within
    cell.precision using the specified grids mesh in AFT Coulomb integrals.
    '''
    # The cutoff estimation
    #     exp(-G^2/(4*(ai+aj))) cs[:,None]*cs * exp(-theta*dr**2)
    # is effective for orbital pairs with large ai+aj. However, its inaccurate
    # for diffuse orbital pairs due to the contribution of lattice summation.
    # Here, we directly evaluate the ft_aopair, and test whether the
    # contributions of G vectors at the edge can be discarded.
    mesh = np.asarray(mesh)
    nx, ny, nz = (mesh+1) // 2
    mask = np.zeros(mesh+2, dtype=bool)
    mask[nx:nx+2,:,:] = True
    mask[:,ny:ny+2,:] = True
    mask[:,:,nz:nz+2] = True
    Gv = cell.get_Gv(mesh + 2)
    Gv = Gv[mask.ravel()]
    ngrids = len(Gv)

    avail_mem = int(get_avail_mem(exclude_memory_pool=True) * .9)
    unit = cell.nao**2*2
    Gblksize = min(ngrids, int(avail_mem/(16*unit)))
    ft_opt = FTOpt(cell)
    ft_opt.rcut = cell.rcut / 2 # reduce accuracy for an estimation of ke_cutoff
    ft_kern = ft_opt.gen_ft_kernel(transform_ao=False)
    pair_max = cp.zeros((cell.nbas, cell.nbas))
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        Gpq = ft_kern(Gv[p0:p1])
        _pair_max = cp.abs(Gpq[0]).max(axis=0)
        _pair_max = condense('absmax', _pair_max, cell.ao_loc)
        pair_max = cp.where(pair_max > _pair_max, pair_max, _pair_max)
    precision = cell.precision * max(1, 1e-2 * cell.vol)
    pair_mask = pair_max < precision
    return pair_mask

def _Ls_to_Bvk_Ts(supmol, kmesh):
    cell = supmol.cell
    # supmol Ts can be mapped to the corresponding Ts in BvK cell
    Ts = np.linalg.solve(cell.lattice_vectors().T, supmol.Ls.T).T
    Ts = np.asarray(Ts.round(), dtype=np.int32)
    Ts_in_bvk = Ts % kmesh
    # Index of each BvK Ts is stored in bvk_address
    bvk_address = np.ravel_multi_index(Ts_in_bvk.T, kmesh)
    return bvk_address

def _double_latsum_in_bvk(supmol, kmesh, kpts):
    cell = supmol.cell
    # Index of each BvK Ts is stored in bvk_address
    bvk_address = cp.asarray(_Ls_to_Bvk_Ts(supmol, kmesh), dtype=np.int32)
    I, ish = divmod(cp.asarray(supmol.bas_mask_idx, dtype=np.int32), cell.nbas)
    bvk_shell_idx = bvk_address[I] * cell.nbas + ish
    Ts_ji_lookup = cp.asarray(double_translation_indices(kmesh), dtype=np.int32)

    bvk_Ls = translation_vectors_for_kmesh(cell, kmesh)
    expLk = cp.exp(1j * asarray(bvk_Ls).dot(asarray(kpts).T))
    return bvk_shell_idx, Ts_ji_lookup, expLk

def _double_latsum_in_supermol(supmol, kpts):
    from gpu4pyscf.pbc.dft.multigrid_v2 import _unique_image_pair
    cell = supmol.cell
    Ts = np.linalg.solve(cell.lattice_vectors().T, supmol.Ls.T).T
    Ts = cp.asarray(Ts.round(), dtype=np.int32)
    double_latsum_Ts, inverse = _unique_image_pair(Ts)
    nimgs = len(supmol.Ls)
    Ts_ji_lookup = cp.asarray(inverse, order='C', dtype=np.int32).reshape(nimgs, nimgs)
    supmol_bas_idx = cp.asarray(supmol.bas_mask_idx, dtype=np.int32)

    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    Ts = cp.asarray(double_latsum_Ts, dtype=np.float64)
    expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
    return supmol_bas_idx, Ts_ji_lookup, expLk

def _dm_cond_from_compressed_dm(supmol, dms):
    '''Largest density matrix elements for each shell-pair within unit cell.
    '''
    cell = supmol.cell
    ao_loc = asarray(cell.ao_loc)
    n_dm, n_Ts, nao = dms.shape[:3]
    Ts_ao_loc = cp.arange(0, n_Ts*nao, nao, dtype=np.int32)[:,None] + ao_loc[:-1]
    Ts_ao_loc = cp.append(Ts_ao_loc.ravel(), np.int32(n_Ts*nao))
    dm_cond = condense('absmax', dms.reshape(n_dm, n_Ts*nao, nao), Ts_ao_loc, ao_loc)
    nbas = cell.nbas
    dm_cond = dm_cond.reshape(n_Ts, nbas, nbas)
    return dm_cond

_Q_COND_BUFSIZE = 30000**2

def _cache_q_cond_and_non0pairs(vhfopt, tile=4, dd_pair_mask=None):
    log = logger.new_logger(vhfopt)
    cell = vhfopt.cell
    supmol = vhfopt.supmol
    omega = -vhfopt.omega

    precision = vhfopt.estimate_cutoff_with_penalty()
    # Adjust precision to improve accuracy for very diffuse orbitals
    s_log_cutoff = q_log_cutoff = math.log(precision)
    #if diffuse_exps.min() < 0.08:
    #    s_log_cutoff += math.log(1e-2)

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
    unit = (li+1)*(lj+1)*2 + (li+1)*(lj+1)*(lij+1) + 6 + nroots*2
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

    raw_bas_idx = cp.empty(nimgs*nbas_cell0, dtype=np.int32)
    raw_bas_idx[bas_mask_idx] = cp.arange(nbas, dtype=np.int32)
    raw_bas_idx = raw_bas_idx.reshape(nimgs, nbas_cell0)
    n_groups = len(l_ctr_bas_loc) - 1
    bas_idx_lookup = []
    for i in range(n_groups):
        ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
        bas_idx = raw_bas_idx[:,ish0:ish1][bas_mask[:,ish0:ish1]]
        bas_idx_lookup.append(cp.asarray(bas_idx, dtype=np.int32, order='C'))

    if dd_pair_mask is None:
        Ecut_mask_ptr = lib.c_null_ptr()
    else:
        Ecut_mask = cp.asarray(dd_pair_mask, dtype=np.int8)
        Ecut_mask_ptr = ctypes.cast(Ecut_mask.data.ptr, ctypes.c_void_p)

    pair_ij_kern = libpbc.PBCsort_pair_ij
    s_kern = libpbc.PBCfill_s_estimator
    q_kern = libpbc.PBCfill_qcond
    pair_ij_kern.restype = ctypes.c_int
    q_kern.restype = ctypes.c_int
    s_kern.restype = ctypes.c_int
    rys_envs = vhfopt.rys_envs
    pair_cache = {}

    n = max(x.size for x in bas_idx_lookup)
    buf_size = min(n**2, _Q_COND_BUFSIZE)
    pair_buf = cp.empty(buf_size, dtype=np.int64)
    s_buf = cp.empty(buf_size, dtype=np.float32)
    split_points = cp.arange(q_log_cutoff, 2., Q_COND_MARGIN)

    def _generate_q_cond(ish, jsh, b0, b1):
        ish = ish[b0:b1]
        nish = len(ish)
        njsh = len(jsh)
        pair_ij = ndarray((nish, njsh), dtype=np.int64, buffer=pair_buf)
        err = pair_ij_kern(
            ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(ish.data.ptr, ctypes.c_void_p),
            ctypes.cast(jsh.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nish), ctypes.c_int(njsh),
            ctypes.c_int(NBAS_MAX), ctypes.c_int(tile))
        if err != 0:
            raise RuntimeError(f'PBCsort_pair_ij kernel failed for group {(i,j)} batch {b0}:{b1}')
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
                     ctypes.c_uint32(pair_ij.size),
                     ctypes.c_double(omega),
                     ctypes.c_int(tril_symmetry),
                     Ecut_mask_ptr)
        if err != 0:
            raise RuntimeError(f'PBCfill_s_estimator kernel failed for group {(i,j)} batch {b0}:{b1}')
        idx = cp.where(s_estimator > s_log_cutoff)[0]
        pair_ij = pair_ij[idx]
        s_estimator = s_estimator[idx]
        q_cond = cp.empty(pair_ij.size, dtype=np.float32)
        if len(pair_ij) > 0:
            err = q_kern(ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                         ctypes.byref(rys_envs), ctypes.c_int(max_shm_size),
                         ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
                         ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                         ctypes.c_uint32(pair_ij.size),
                         ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError('PBCfill_qcond kernel failed for group {(i,j)} batch {b0}:{b1}')
        return pair_ij, q_cond, s_estimator

    for i in range(n_groups):
        for j in range(i+1):
            nish_cell0 = cell.l_ctr_counts[i]
            ish = bas_idx_lookup[i]
            jsh = bas_idx_lookup[j]
            pair_ij, q_cond_ij, s_estimator_ij = _generate_q_cond(ish, jsh, 0, nish_cell0)
            idx = cp.argsort(q_cond_ij)[::-1]
            # pairs with negligible q_cond_ij are excluded
            idx = idx[:int((q_cond_ij > q_log_cutoff).sum())]
            pair_ij = pair_ij[idx]
            q_cond_ij = q_cond_ij[idx]
            s_estimator_ij = s_estimator_ij[idx]

            nish = len(ish)
            njsh = len(jsh)
            # For large unit cell, pair_ij(nish,njsh) may easiy exceed available
            # memory, process ish in small batches.
            batch_size = max(1, buf_size // (njsh*tile)) * tile
            batch_locs = list(range(0, nish, batch_size)) + [nish]

            results = [_generate_q_cond(ish, jsh, b0, b1)
                       for b0, b1 in zip(batch_locs[:-1], batch_locs[1:])]
            if len(results) == 1:
                pair_kl, q_cond_kl, s_estimator_kl = results[0]
            else:
                pair_kl = cp.hstack([x[0] for x in results])
                q_cond_kl = cp.hstack([x[1] for x in results])
                s_estimator_kl = cp.hstack([x[2] for x in results])
            idx = _group_by_split_points(q_cond_kl, split_points)
            pair_kl = pair_kl[idx]
            q_cond_kl = q_cond_kl[idx]
            s_estimator_kl = s_estimator_kl[idx]

            log.debug1('(%d,%d) len(pair_ij) = %d, len(pair_kl) = %d',
                       i, j, pair_ij.size, pair_kl.size)
            pair_cache[i,j] = (pair_ij, q_cond_ij, s_estimator_ij,
                               pair_kl, q_cond_kl, s_estimator_kl)
    return pair_cache

def _guess_omega(cell, kpts=None):
    if kpts is None or kpts.ndim == 1:
        nkpts = 1
    else:
        nkpts = len(kpts)
    nao = cell.nao_nr(cart=True)
    ng = int(5e4/(nao*nkpts**.65))
    ng = (max(3, ng) // 2) * 2 + 1
    if ng >= 11:
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, OMEGA)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh[mesh>ng] = ng
    else:
        mesh = [ng] * 3
    ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh)
    ke_cutoff = ke_cutoff[:cell.dimension].min()
    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff)

    OMEGA_MIN = 0.08
    if omega < OMEGA_MIN:
        logger.warn(cell, 'omega=%g smaller than the required minimal value %g. '
                    'Set omega to %g', omega, OMEGA_MIN, OMEGA_MIN)
        omega = OMEGA_MIN
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
    return omega, ke_cutoff, mesh

def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
    '''Energy cutoff for AFTDF to converge attenuated Coulomb in moment space
    '''
    if precision is None:
        precision = cell.precision
    # Errors are dominated by the Coulomb integrals of the most compact density.
    # In this case, the error estimation can be approximated as
    # sum_(G^2>Ecut) 4*pi/G^2 exp(-G^2/(4*omega^2))
    #     ~ 16\pi^2 \int_sqrt(2*Ecut)^inf exp(-G^2/(4*omega^2)) dG
    #     < 16\pi^2 * 2*omega^2 / sqrt(2*Ecut) exp(-Ecut/(2*omega^2))
    exps, cs = extract_pgto_params(cell, 'compact')
    exp_max = exps.max()
    theta = 1./(1./(4*exp_max) + omega**-2)
    fac = 16*np.pi**2/cell.vol * 2*theta / precision
    Ecut = 20.
    Ecut = math.log(fac / (2*Ecut)**.5) * 2*theta
    Ecut = math.log(fac / (2*Ecut)**.5) * 2*theta
    return Ecut

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulomb given energy cutoff
    '''
    if precision is None:
        precision = cell.precision
    # estimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
    # enough to converge the 2-electron integrals. A penalty term here is to
    # reduce the error in integrals
    exps, cs = extract_pgto_params(cell, 'compact')
    exp_max = exps.max()
    fac = 16*np.pi**2/cell.vol / (2*ke_cutoff)**.5 / precision
    omega = 0.5
    theta = 1./(1./(4*exp_max) + omega**-2)
    omega = (.5 * ke_cutoff / math.log(fac*2*theta))**.5
    theta = 1./(1./(4*exp_max) + omega**-2)
    omega = (.5 * ke_cutoff / math.log(fac*2*theta))**.5
    return omega

def _group_by_split_points(q_cond, split_points):
    # Use np.digitize to assign each value to a bin
    bin_indices = cp.searchsorted(split_points, q_cond)
    num_bins = len(split_points)
    # Collect values. exclude the first one, as their q_cond values are
    # sufficiently small
    subsets = [cp.where(bin_indices == i)[0] for i in range(1, num_bins+1)]
    # Sorting the values, from large to small. This allows the integral
    # screening testing terminating early.
    return cp.hstack(subsets[::-1])

def _get_vk_wcoulG_and_SR(cell, kpt, kpts, exxdiv, mesh, Gv, Gv_weight,
                          rsjk_omega, omega, lr_factor, sr_factor):
    # wcoulG is Coulomb kernel for the aggregated operator
    # lr_factor * erf(|omega|r12)/r12 + sr_factor * erfc(|omega|r12)/r12.
    wcoulG = get_coulG(cell, kpt, exx=exxdiv, mesh=mesh, Gv=Gv,
                       wrap_around=True, omega=0, kpts=kpts)
    if lr_factor == sr_factor:
        wcoulG *= lr_factor * Gv_weight
    else:
        coulG_LR = get_coulG(cell, kpt, exx=exxdiv, mesh=mesh, Gv=Gv,
                             wrap_around=True, omega=omega, kpts=kpts)
        wcoulG -= coulG_LR
        wcoulG *= sr_factor * Gv_weight
        coulG_LR *= lr_factor * Gv_weight
        wcoulG += coulG_LR

    # This coulG_SR attemps to remove the low-Ecut part of get_k_sr integrals
    wcoulG_SR = get_coulG(cell, kpt, exx=None, mesh=mesh, Gv=Gv,
                          wrap_around=True, omega=-rsjk_omega, kpts=kpts)
    if is_zero(Gv[0]+kpt):
        wcoulG_SR[0] += np.pi / rsjk_omega**2
    wcoulG_SR *= sr_factor * Gv_weight
    return wcoulG, wcoulG_SR

def _generate_shl_pairs(ft_opt, dd_bas_idx):
    cell = ft_opt.cell
    bvk_ncells = len(ft_opt.bvkmesh_Ls)
    nbas = cell.nbas
    mask_exclude_dd = cp.zeros((nbas*bvk_ncells*nbas), dtype=bool)
    _idx = cp.hstack([bas_ij for bas_ij in ft_opt.bas_ij_cache.values()])
    mask_exclude_dd[_idx] = True
    mask_exclude_dd = mask_exclude_dd.reshape(nbas, bvk_ncells, nbas)
    ish, jsh = divmod(asarray(dd_bas_idx), nbas)
    mask_exclude_dd[ish,:,jsh] = False
    mask_exclude_dd = mask_exclude_dd.ravel()[_idx].get()

    img_offsets = ft_opt.img_offsets.get()
    img_counts = img_offsets[1:] - img_offsets[:-1]
    bas_ij_idx = []
    bas_ij_img_idx = []
    shl_pair_offsets = []
    sp0 = sp1 = 0
    p0 = p1 = 0
    for (i, j), bas_ij in ft_opt.bas_ij_cache.items():
        p0, p1 = p1, p1 + len(bas_ij)
        img_counts_ij = img_counts[p0:p1]
        mask = np.repeat(mask_exclude_dd[p0:p1], img_counts_ij)
        bas_ij = np.repeat(bas_ij.get(), img_counts_ij)[mask]
        bas_ij_idx.append(asarray(bas_ij, dtype=np.int32))
        img_idx = ft_opt.img_idx[img_offsets[p0]:img_offsets[p1]]
        bas_ij_img_idx.append(img_idx[mask])
        sp0, sp1 = sp1, sp1 + len(bas_ij)
        shl_pair_offsets.append(np.arange(sp0, sp1, 128, dtype=np.int32))
    shl_pair_offsets.append(np.int32(sp1))
    bas_ij_idx = cp.hstack(bas_ij_idx, dtype=np.int32)
    bas_ij_img_idx = cp.hstack(bas_ij_img_idx, dtype=np.int32)
    shl_pair_offsets = cp.hstack(shl_pair_offsets, dtype=np.int32)
    return bas_ij_idx, bas_ij_img_idx, shl_pair_offsets

def _bas_recontract_ft_pair(cell, Gpq, out=None, buf=None):
    pqG = Gpq.transpose(0,2,3,1)
    assert pqG.flags.c_contiguous
    tmp = cell.apply_CT_dot(pqG, axis=1, out=buf)
    out = cell.apply_CT_dot(tmp, axis=2, out=out)
    return out.transpose(0,3,1,2)
