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
from pyscf.pbc.scf.rsjk import estimate_ke_cutoff_for_omega
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    condense, transpose_sum, dist_matrix, contract, asarray, ndarray)
from gpu4pyscf.gto.mole import (
    group_basis, groupby, extract_pgto_params, SortedCell)
from gpu4pyscf.scf.jk import (
    libvhf_rys, RysIntEnvVars, _scale_sp_ctr_coeff,
    _nearest_power2, apply_coeff_C_mat_CT, apply_coeff_CT_mat_C,
    PTR_BAS_COORD, LMAX, QUEUE_DEPTH, SHM_SIZE, GOUT_WIDTH, THREADS)
from gpu4pyscf.pbc.df.ft_ao import libpbc, most_diffuse_pgto, PBCIntEnvVars
from gpu4pyscf.pbc.df.fft import _check_kpts
from gpu4pyscf.pbc.df.fft_jk import _format_dms
from gpu4pyscf.pbc.dft.multigrid_v2 import _unique_image_pair
from gpu4pyscf.pbc.tools.pbc import get_coulG, probe_charge_sr_coulomb
from gpu4pyscf.grad.rhf import _ejk_quartets_scheme
from gpu4pyscf.pbc.gto import int1e

__all__ = [
    'get_k',
]

libpbc.PBC_build_k.restype = ctypes.c_int
libpbc.PBC_build_k_init(ctypes.c_int(SHM_SIZE))
libpbc.PBC_build_jk_ip1_init(ctypes.c_int(SHM_SIZE))
libpbc.PBC_per_atom_jk_ip1.restype = ctypes.c_int
libpbc.PBC_jk_strain_deriv.restype = ctypes.c_int

DD_CACHE_MAX = 101250 * (SHM_SIZE//48000)
OMEGA = 0.4
NBAS_MAX = 1048576

def get_k(cell, dm, hermi=0, kpts=None, kpts_band=None, omega=None, vhfopt=None,
          lr_factor=None, sr_factor=None, exxdiv=None, verbose=None):
    '''Compute K matrix
    '''
    if vhfopt is None:
        vhfopt = PBCJKMatrixOpt(cell, omega)
    else:
        assert isinstance(vhfopt, PBCJKMatrixOpt)
    if vhfopt.supmol is None:
        if omega is not None and omega != 0:
            vhfopt.omega = omega
        vhfopt.build(kpts, verbose=verbose)
    else:
        assert omega is None or omega == 0 or omega == vhfopt.omega

    vk = None
    if sr_factor != 0:
        vk = vhfopt._get_k_sr(dm, hermi, kpts, kpts_band,
                              exxdiv=exxdiv, verbose=verbose)
        if sr_factor is not None:
            vk *= sr_factor

    if lr_factor != 0:
        vk_lr = vhfopt._get_k_lr(dm, hermi, kpts, kpts_band,
                                 exxdiv=exxdiv, verbose=verbose)
        if lr_factor is not None:
            vk_lr *= lr_factor
        if vk is None:
            vk = vk_lr
        else:
            vk += vk_lr
    elif vk is None:
        vk = 0
    return vk

class PBCJKMatrixOpt:

    def __init__(self, cell, omega=None):
        self.cell = cell
        self.verbose = cell.verbose
        self.stdout = cell.stdout

        self.omega = omega
        self.mesh = None
        self.supmol = None

        # Attributes required by AFTDF functions
        self.time_reversal_symmetry = True

        # Hold cache on GPU devices
        self._rys_envs = {}

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('_rys_envs', '_q_cond', '_s_estimator'))

    def build(self, kpts=None, verbose=None):
        log = logger.new_logger(self, verbose)
        cput0 = log.init_timer()
        cell = self.cell = SortedCell.from_cell(
            self.cell, decontract=True, diffuse_cutoff=0.2)
        lmax = cell.uniq_l_ctr[:,0].max()
        if lmax > LMAX:
            raise NotImplementedError('basis set with h functions')

        if self.omega is None or self.omega == 0:
            self.omega = _guess_omega(cell, kpts)
        if self.mesh is None:
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)

        cell.omega = -self.omega
        log.debug1('PBCJKMatrixOpt.build: omega = %g mesh = %s', self.omega, self.mesh)

        # FIXME: should the supmol be regrouped based on l?
        self.supmol = ExtendedMole.from_cell(cell, self.omega)

        self.bas_pair_cache = _cache_q_cond_and_non0pairs(self, tile=6)
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
        double_lat_sum_penalty = max(1, (50/(exp_min*lat_unit**2))**3)
        cutoff = precision*1e-1 / lattice_sum_factor / double_lat_sum_penalty
        logger.debug1(cell, 'rsjk integral theta=%g cutoff=%g '
                      'lattice_sum_factor=%g double_lat_sum_penalty=%g',
                      theta, cutoff, lattice_sum_factor, double_lat_sum_penalty)
        return cutoff

    def _get_k_sr(self, dm, hermi, kpts=None, kpts_band=None, exxdiv=None, verbose=None):
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

        if kpts is None:
            kpts = np.zeros((1, 3))
        else:
            kpts = kpts.reshape(-1, 3)
        is_gamma_point = is_zero(kpts)
        is_real = True
        if is_gamma_point:
            assert dms.dtype == np.float64
            nkpts = 1
            ao_loc = asarray(cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            if hermi == 0:
                # Wrap the triu contribution to tril
                dm_cond = dm_cond + dm_cond.T
            # Add the dimension for kpts
            dms = dms[:,None,:,:]
            n_dm = len(dms)
        else:
            scaled_kpts = kpts.dot(cell.lattice_vectors().T)
            Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
            expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
            nkpts = expLk.shape[1]
            dms = dms.reshape(-1, nkpts, nao, nao)
            n_dm = len(dms)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            expLk = None
            # Are dms always real for super-mol?
            if abs(dms.imag).max() < cell.precision*5e2:
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

        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(supmol, 'diffuse')

        uniq_l_ctr = cell.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = np.append(0, np.cumsum(cell.l_ctr_counts))
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        # TODO: i >= k if hermi == 1
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
                dms = cp.vstack([dms, dms[:,::-1].transpose(0,1,3,2)])
            dm_counts = len(dms)

            _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
            bas_pair_cache = {k: [cp.asarray(x) for x in v]
                              for k, v in self.bas_pair_cache.items()}
            bas_mask_idx = cp.asarray(supmol.bas_mask_idx)

            nimgs = len(supmol.Ls)
            if is_gamma_point:
                Ts_ji_lookup = cp.zeros_like(supmol.Ts_ji_lookup)
                nimgs_uniq_pair = 1
            else:
                Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup)
                nimgs_uniq_pair = len(supmol.double_latsum_Ts)
            vk = cp.zeros(dms.shape)

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+3, dtype=np.int64)

            timing_counter = Counter()
            kern_counts = 0
            kern = libpbc.PBC_build_k
            rys_envs = self.rys_envs
            omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, _, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                _, pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                err = kern(
                    ctypes.cast(vk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(dm_counts), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(SHM_SIZE),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(Ts_ji_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nimgs), ctypes.c_int(nimgs_uniq_pair),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(cell.nbas),
                    supmol._bas.ctypes, ctypes.c_double(omega))
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'PBC_build_k kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()

            if kpts_band is not None:
                raise NotImplementedError

            if hermi == 0:
                n = dm_counts // 2
                vk[:n] += vk[n:,::-1].transpose(0,1,3,2)
                vk = vk[:n]

            if not is_gamma_point:
                scaled_kpts = kpts.dot(cell.lattice_vectors().T)
                Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
                expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
                expLkz = expLk.view(np.float64).reshape(nimgs_uniq_pair, nkpts, 2)
                vk = contract('sLmn,Lkz->skmnz', vk, expLkz)
                vk = cp.asarray(vk, order='C').view(np.complex128)[:,:,:,:,0]

            if not is_real:
                vk = vk[:n_dm] + vk[n_dm:] * 1j
            return vk, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)

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

        vk = vk.reshape(-1,nao,nao)
        if hermi == 1:
            vk = transpose_sum(vk)
        vk = cell.apply_CT_mat_C(vk)

        # In FFTDF.get_jk(), the SR integrals at G=0 are added back to K matrix
        # by the Ewald correction. When the vk_sr is evaluated in real space,
        # the G=0 component is included in vk_sr. In vk_lr, only the long-range
        # Coulomb correction needs to be considered in the exxdiv='ewald'.
        if ((cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            # difference associated to the G=0 term between the real space
            # integrals and the AFT integrals
            vk = vk.reshape(n_dm, nkpts, nao_orig, nao_orig)
            dms = dm.reshape(n_dm, nkpts, nao_orig, nao_orig)
            omega = self.omega
            if exxdiv == 'ewald':
                # probe_charge_sr_coulomb equals to -2*ewovrl.
                # This term rapidly decays to 0 for large k-mesh. In the
                # FFTDF.get_jk based implementation, this contribution is
                # included in the short-range part.
                wcoulG_SR_at_G0 = probe_charge_sr_coulomb(cell, omega, kpts)
            else:
                # Remove the G=0 contribution to match the output of FFTDF.get_jk().
                wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
            s = int1e.int1e_ovlp(cell, kpts)
            for i in range(n_dm):
                for k in range(nkpts):
                    vk[i,k] -= s[k].dot(dms[i,k]).dot(s[k]) * wcoulG_SR_at_G0

        if not is_gamma_point:
            weight = 1. / nkpts
            vk *= weight

        if kpts_band is None:
            vk = vk.reshape(dm.shape)
        else:
            raise NotImplementedError
        return vk

    def _get_k_lr(self, dm, hermi, kpts=None, kpts_band=None, exxdiv=None,
                  verbose=None):
        from gpu4pyscf.pbc.df.aft_jk import get_k_kpts
        cell = self.cell
        assert cell.dimension == 3
        kpts, is_single_kpt = _check_kpts(kpts, dm)
        if is_single_kpt:
            kpts = kpts[0]
        return get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv=exxdiv)

    def weighted_coulG(self, kpt=None, exx=None, mesh=None, omega=None, kpts=None):
        '''weighted LR Coulomb kernel. Mimic AFTDF.weighted_coulG'''
        if mesh is None:
            mesh = self.mesh
        cell = self.cell
        omega = self.omega
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = get_coulG(cell, kpt, exx=None, mesh=mesh, Gv=Gv,
                          wrap_around=True, omega=omega, kpts=kpts)
        coulG *= kws
        if kpt is None or not is_zero(kpt):
            return coulG

        if exx == 'ewald':
            Nk = len(kpts)
            # In the full-range Coulomb, the ewald correction for get_k_lr is
            #     +Nk*pbctools.madelung(cell, kpts) - np.pi / omega**2 * kws - probe_charge_sr_coulomb
            # The last two terms are included in the get_k_sr. The second term
            # (np.pi/omega**2) removes the contribution of the SR integrals at G=0.
            #
            # pbctools.madelung(cell, kpts) includes three terms: -2*ewovrl, -2*ewself and -2*ewg.
            # ewself is the sum of ewself_lr_point_charge and ewself_sr_at_G0.
            # This correction is identical to madelung(cell, kpts, omega=omega),
            # which gives -2*(ewself_lr_point_charges + ewg) .
            # ewself_sr_at_G0 in ewovrl cancels out the second term (np.pi/omega**2);
            # -2*ewovrl cancels out the last term (probe_charge_sr_coulomb).
            coulG[0] += Nk*pbctools.madelung(cell, kpts, omega=omega)
        return coulG

    def _get_ejk_sr_ip1(self, dm, kpts=None, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
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

        if kpts is None:
            kpts = np.zeros((1, 3))
        else:
            kpts = kpts.reshape(-1, 3)
        is_gamma_point = is_zero(kpts)
        if is_gamma_point:
            assert dms.dtype == np.float64
            nkpts = 1
            ao_loc = asarray(cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            # Add the dimension for kpts
            dms = dms[:,None,:,:]
        else:
            scaled_kpts = kpts.dot(cell.lattice_vectors().T)
            Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
            expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
            nkpts = expLk.shape[1]
            dms = dms.reshape(-1, nkpts, nao, nao)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            assert abs(dms.imag).max() < 1e-6
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
            bas_mask_idx = cp.asarray(supmol.bas_mask_idx)

            nimgs = len(supmol.Ls)
            if is_gamma_point:
                Ts_ji_lookup = cp.zeros_like(supmol.Ts_ji_lookup)
                nimgs_uniq_pair = 1
            else:
                Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup)
                nimgs_uniq_pair = len(supmol.double_latsum_Ts)
            ejk = cp.zeros((cell.natm, 3))

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int64)
            dd_pool = cp.empty((workers, DD_CACHE_MAX), dtype=np.float64)

            t1 = log.timer_debug1(f'ejk_sr initialization on Device {device_id}', *t0)
            timing_counter = Counter()
            kern_counts = 0
            kern = libpbc.PBC_per_atom_jk_ip1
            rys_envs = self.rys_envs
            omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, _, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                _, pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = _ejk_quartets_scheme(supmol, uniq_l_ctr[[i, j, k, l]])
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(Ts_ji_lookup.data.ptr, ctypes.c_void_p),
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
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'PBC_build_jk_ip1 kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()
            return ejk, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)

        kern_counts = 0
        timing_collection = Counter()
        ejk_dist = []
        for ejk, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            ejk_dist.append(ejk)

        log = logger.new_logger(cell, verbose)
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        ejk = multi_gpu.array_reduce(ejk_dist, inplace=True)
        ejk = ejk.get()

        if ((cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
            # difference associated to the G=0 term between the real space
            # integrals and the AFT integrals
            dms = dm.reshape(n_dm, nkpts, nao_orig, nao_orig)
            omega = self.omega
            wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
            if exxdiv == 'ewald':
                wcoulG_for_k = probe_charge_sr_coulomb(cell, omega, kpts)
            else:
                wcoulG_for_k = wcoulG_SR_at_G0
            s0 = int1e.int1e_ovlp(cell, kpts)
            s1 = int1e.int1e_ipovlp(cell, kpts)
            j_dm = cp.einsum('kij,nkji->', s0, dms)
            j_dm = dms.sum(axis=0) * (j_factor * j_dm * wcoulG_SR_at_G0)
            k_dm = contract('nkpq,kqr->nkpr', dms, s0)
            k_dm = contract('nkpr,nkrs->kps', k_dm, dms)
            if n_dm == 1: # RHF
                k_dm *= .5 * k_factor * wcoulG_for_k
            else:
                k_dm *= k_factor * wcoulG_for_k
            ejk += contract_h1e_dm(cell.cell, s1, j_dm-k_dm, hermi=1) * .5

        if not is_gamma_point:
            ejk *= 1. / nkpts**2
        return ejk

    def _get_ejk_lr_ip1(self, dm, kpts=None, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
        '''Compute the derivatives of the long-range part of the aggregated
        J/K contribution. The aggregated J/K contribution is given by
        j_factor*J-k_factor*K/2 for RHF and j_factor*J-k_factor*K for UHF.
        '''
        from gpu4pyscf.pbc.df.aft_jk import get_ej_ip1, get_ek_ip1
        cell = self.cell
        assert cell.dimension == 3
        dm = _format_dms(dm, kpts)
        n_dm = len(dm)
        if kpts is None:
            kpts = np.zeros((1,3))
        else:
            kpts = kpts.reshape(-1, 3)
        ej = ek = 0
        if j_factor != 0:
            ej = get_ej_ip1(self, dm, kpts)
            ej *= j_factor
        if k_factor != 0:
            # RHF energy is computed as J - 1/2 K
            if n_dm == 1: # RHF or KRHF
                k_factor *= .5
            ek = get_ek_ip1(self, dm, kpts, exxdiv=exxdiv)
            ek *= k_factor
        return ej - ek

    def _get_ejk_sr_strain_deriv(self, dm, kpts=None, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
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

        if kpts is None:
            kpts = np.zeros((1, 3))
        else:
            kpts = kpts.reshape(-1, 3)
        is_gamma_point = is_zero(kpts)
        if is_gamma_point:
            assert dms.dtype == np.float64
            nkpts = 1
            ao_loc = asarray(cell.ao_loc)
            dms = cp.asarray(dms, order='C')
            dm_cond = condense('absmax', dms, ao_loc)
            # Add the dimension for kpts
            dms = dms[:,None,:,:]
        else:
            scaled_kpts = kpts.dot(cell.lattice_vectors().T)
            Ts = cp.asarray(supmol.double_latsum_Ts, dtype=np.float64)
            expLk = cp.exp(1j * Ts.dot(asarray(scaled_kpts).T))
            nkpts = expLk.shape[1]
            dms = dms.reshape(-1, nkpts, nao, nao)
            dms = contract('skpq,Lk->sLpq', dms, expLk)
            # Are dms always real for super-mol?
            assert abs(dms.imag).max() < 1e-6
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
            bas_mask_idx = cp.asarray(supmol.bas_mask_idx)

            nimgs = len(supmol.Ls)
            if is_gamma_point:
                Ts_ji_lookup = cp.zeros_like(supmol.Ts_ji_lookup)
                nimgs_uniq_pair = 1
            else:
                Ts_ji_lookup = cp.asarray(supmol.Ts_ji_lookup)
                nimgs_uniq_pair = len(supmol.double_latsum_Ts)
            ejk = cp.zeros((cell.natm, 3))
            sigma = cp.zeros((3, 3))

            workers = gpu_specs['multiProcessorCount']
            pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int64)
            dd_pool = cp.empty((workers, DD_CACHE_MAX), dtype=np.float64)

            t1 = log.timer_debug1(f'ejk_sr_strain_deriv initialization on Device {device_id}', *t0)
            timing_counter = Counter()
            kern_counts = 0
            kern = libpbc.PBC_jk_strain_deriv
            rys_envs = self.rys_envs
            omega = -self.omega

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, _, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                _, pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                scheme = _ejk_quartets_scheme(supmol, uniq_l_ctr[[i, j, k, l]])
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                    ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    ctypes.byref(rys_envs), (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(Ts_ji_lookup.data.ptr, ctypes.c_void_p),
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
                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'PBC_jk_strain_deriv kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()
            return ejk, sigma, kern_counts, timing_counter

        results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        dms = None

        kern_counts = 0
        timing_collection = Counter()
        ejk_dist = []
        sigma_dist = []
        for ejk, sigma, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            ejk_dist.append(ejk)
            sigma_dist.append(sigma)

        log = logger.new_logger(cell, verbose)
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        ejk = multi_gpu.array_reduce(ejk_dist, inplace=True)
        sigma = multi_gpu.array_reduce(sigma_dist, inplace=True)
        sigma = sigma.get()
        sigma *= 2 / nkpts**2
        if not is_gamma_point:
            ejk *= 1. / nkpts**2
        ejk = ejk.get()

        if ((cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
            # difference associated to the G=0 term between the real space
            # integrals and the AFT integrals
            dm0 = dm.reshape(n_dm, nkpts, nao_orig, nao_orig)
            omega = self.omega
            wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
            if exxdiv == 'ewald':
                wcoulG_for_k = probe_charge_sr_coulomb(cell, omega, kpts)
            else:
                wcoulG_for_k = wcoulG_SR_at_G0

            s0 = int1e.int1e_ovlp(cell, kpts)
            s1 = int1e.int1e_ipovlp(cell, kpts)
            nelectron = cp.einsum('kij,nkji->', s0, dm0).real.get() / nkpts
            j_dm = dm0.sum(axis=0) * (j_factor * nelectron * wcoulG_SR_at_G0)
            k_dm = contract('nkpq,kqr->nkpr', dm0, s0)
            k_dm = contract('nkpr,nkrs->kps', k_dm, dm0)
            ej_G0 = .5 * cp.einsum('kij,kji->', s0, j_dm).real.get() / nkpts
            ek_G0 = .5 * cp.einsum('kij,kji->', s0, k_dm).real.get() * k_factor / nkpts**2
            if n_dm == 1: # RHF
                ek_G0 *= .5
                k_dm *= .5 * k_factor * wcoulG_for_k / nkpts
            else:
                k_dm *= k_factor * wcoulG_for_k / nkpts
            ejk_G0 = contract_h1e_dm(cell.cell, s1, j_dm-k_dm, hermi=1) * .5
            ejk += ejk_G0 / nkpts

            # Response of the overlap integrals in Tr(S D S D)
            int1e_opt = int1e._Int1eOpt(cell, 1)
            sigma -= int1e_opt.get_ovlp_strain_deriv(j_dm, kpts)
            sigma += int1e_opt.get_ovlp_strain_deriv(k_dm, kpts)
            # Response of 1/cell.vol within the G=0 term of the coulG_SR
            sigma += ej_G0 * np.eye(3)
            sigma -= wcoulG_SR_at_G0 * ek_G0 * np.eye(3)
            if exxdiv == 'ewald':
                from pyscf.pbc.tools.pbc import madelung
                from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells
                scaled_kpts = kpts.dot(cell.lattice_vectors().T)
                ewald_G0_response = np.empty((3,3))
                disp = max(1e-5, (cell.precision*.1)**.5)
                for i in range(3):
                    for j in range(i+1):
                        cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
                        kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
                        kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
                        e1 = nkpts * madelung(cell1, kpts1, omega=-omega)
                        e2 = nkpts * madelung(cell2, kpts2, omega=-omega)
                        ewald_G0_response[j,i] = ewald_G0_response[i,j] = (e1-e2)/(2*disp)
                ewald_G0_response *= ek_G0
                sigma -= ewald_G0_response

        return sigma

    def _get_ejk_lr_strain_deriv(self, dm, kpts=None, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
        '''Compute the strain derivatives of the long-range part of the
        aggregated J/K contribution. The aggregated J/K contribution is given by
        j_factor*J-k_factor*K/2 for RHF and j_factor*J-k_factor*K for UHF.
        '''
        from gpu4pyscf.pbc.df.aft_jk import get_ej_strain_deriv, get_ek_strain_deriv
        cell = self.cell
        assert cell.dimension == 3
        dm = _format_dms(dm, kpts)
        n_dm = len(dm)
        ej = ek = 0
        if j_factor != 0:
            ej = get_ej_strain_deriv(self, dm, kpts, omega=self.omega)
            ej *= j_factor
        if k_factor != 0:
            # RHF energy is computed as J - 1/2 K
            if n_dm == 1: # RHF or KRHF
                k_factor *= .5
            ek = get_ek_strain_deriv(self, dm, kpts, exxdiv=exxdiv,
                                     omega=self.omega)
            ek *= k_factor
        return ej - ek

    def jk_energy_per_atom(self, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                           exxdiv=None, with_long_range=True, verbose=None):
        raise

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
        supmol.Ts_ji_lookup = cp.asarray(inverse, order='C', dtype=np.int32).reshape(nimgs, nimgs)
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
    fac /= aij**(li+1.5) * akl**(lk+1.5) * aj**lj * al**ll
    fac *= fl / precision

    vol = cell.vol
    lat_unit = vol**(1./3)
    rad = cell.rcut / lat_unit + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*(cell.rcut)/(vol*theta) + surface
    fac *= lattice_sum_factor * 10

    r0 = cell.rcut
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
    rcut = r0
    return rcut

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

def _cache_q_cond_and_non0pairs(vhfopt, tile=4):
    cell = vhfopt.cell
    supmol = vhfopt.supmol
    omega = -vhfopt.omega

    precision = vhfopt.estimate_cutoff_with_penalty()
    diffuse_exps = extract_pgto_params(cell, 'diffuse')[0]
    # Adjust precision to improve accuracy for very diffuse orbitals
    s_log_cutoff = q_log_cutoff = math.log(precision)
    if diffuse_exps.min() < 0.08:
        s_log_cutoff += math.log(1e-2)

    diffuse_idx = groupby(cell._bas[:,gto.ATOM_OF], diffuse_exps, 'argmin')
    diffuse_exps_per_atom = cp.array(diffuse_exps[diffuse_idx], dtype=np.float32)

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
    raw_bas_idx = cp.empty(nimgs*nbas_cell0, dtype=np.int32)
    raw_bas_idx[bas_mask_idx] = cp.arange(nbas, dtype=np.int32)
    raw_bas_idx = raw_bas_idx.reshape(nimgs, nbas_cell0)
    n_groups = len(l_ctr_bas_loc) - 1
    bas_idx_lookup = []
    for i in range(n_groups):
        ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
        bas_idx = raw_bas_idx[:,ish0:ish1][bas_mask[:,ish0:ish1]]
        bas_idx_lookup.append(cp.asarray(bas_idx, dtype=np.int32, order='C'))

    n = max(x.size for x in bas_idx_lookup)
    pair_buf = cp.empty(n**2, dtype=np.int64)

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
            pair_ij = ndarray((nish, njsh), dtype=np.int64, buffer=pair_buf)
            pair_ij[:]=0
            err = pair_ij_kern(
                ctypes.cast(pair_ij[:nish_cell0].data.ptr, ctypes.c_void_p),
                ctypes.cast(ish[:nish_cell0].data.ptr, ctypes.c_void_p),
                ctypes.cast(jsh.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nish_cell0), ctypes.c_int(njsh),
                ctypes.c_int(tile))
            if err:
                raise RuntimeError('PBCsort_pair_ij kernel failed')
            if nish_cell0 < nish:
                err = pair_ij_kern(
                    ctypes.cast(pair_ij[nish_cell0:].data.ptr, ctypes.c_void_p),
                    ctypes.cast(ish[nish_cell0:].data.ptr, ctypes.c_void_p),
                    ctypes.cast(jsh.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nish-nish_cell0), ctypes.c_int(njsh),
                    ctypes.c_int(tile))
                if err:
                    raise RuntimeError('PBCsort_pair_ij kernel failed')

            pair_ij = pair_ij.ravel()
            s_estimator = cp.empty(pair_ij.size, dtype=np.float32)
            err = s_kern(ctypes.cast(s_estimator.data.ptr, ctypes.c_void_p),
                         ctypes.byref(rys_envs),
                         ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
                         ctypes.cast(bas_mask_idx.data.ptr, ctypes.c_void_p),
                         ctypes.cast(diffuse_exps_per_atom.data.ptr, ctypes.c_void_p),
                         ctypes.c_float(s_log_cutoff),
                         ctypes.c_int(nbas_cell0),
                         ctypes.c_int(len(diffuse_exps_per_atom)),
                         ctypes.c_int(len(pair_ij)),
                         ctypes.c_double(omega))
            if err:
                raise RuntimeError('PBCfill_s_estimator kernel failed')
            idx = cp.where(s_estimator > s_log_cutoff)[0]
            pair_ij = pair_ij[idx]
            s_estimator = s_estimator[idx]
            if len(pair_ij) == 0:
                pair_kl = pair_ij
                q_cond = s_estimator
                pair_cache[i,j] = (pair_ij, pair_kl, q_cond, s_estimator)
                continue

            q_cond = cp.empty(pair_ij.size, dtype=np.float32)
            err = q_kern(ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                         ctypes.byref(rys_envs), ctypes.c_int(max_shm_size),
                         ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
                         ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                         ctypes.c_int(len(pair_ij)),
                         ctypes.c_double(omega))
            if err:
                raise RuntimeError('PBCfill_qcond kernel failed')
            idx = cp.where(q_cond > q_log_cutoff)[0]
            s_estimator = s_estimator[idx]
            q_cond = q_cond[idx]
            pair_kl = pair_ij[idx]
            i_cell0_count = (pair_kl < nbas_cell0 * NBAS_MAX).sum()
            pair_ij = pair_kl[:i_cell0_count]
            pair_cache[i,j] = (pair_ij, pair_kl, q_cond, s_estimator)
    return pair_cache

def _guess_omega(cell, kpts=None):
    if kpts is None:
        nkpts = 1
    else:
        nkpts = len(kpts)
    nao = cell.nao
    bvk_nao = nao * nkpts
    ng = int(1e3/bvk_nao**.5)
    ng = (max(3, ng) // 2) * 2 + 1
    if ng >= 11:
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, OMEGA)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh[mesh<ng] = ng
    else:
        mesh = [ng] * 3
    ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh)
    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff.max())
    return omega

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulomb given energy cutoff
    '''
    if precision is None:
        precision = cell.precision
#    # estimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
#    # enough to converge the 2-electron integrals. A penalty term here is to
#    # reduce the error in integrals
#    precision *= 1e-2
#    kmax = (ke_cutoff*2)**.5
#    log_rest = np.log(precision / (16*np.pi**2 * kmax**lmax))
#    omega = (-.5 * ke_cutoff / log_rest)**.5
#    return omega

    ai = np.hstack(cell.bas_exps()).max()
    aij = ai * 2
    fac = 32*np.pi**2 / precision
    omega = max(.4, ai**.5)
    theta = 1./(1./ai + omega**-2)
    omega2 = 1./(np.log(fac * theta/ (2*ke_cutoff) + 1.)*2/ke_cutoff - 1./aij)
    if omega2 > 0:
        theta = 1./(1./ai + 1./omega2)
        omega2 = 1./(np.log(fac * theta/ (2*ke_cutoff) + 1.)*2/ke_cutoff - 1./aij)
        omega = omega2**.5
    OMEGA_MIN = 0.08
    if omega < OMEGA_MIN:
        logger.warn(cell, 'omega=%g smaller than the required minimal value %g. '
                    'Set omega to %g', omega2, OMEGA_MIN, OMEGA_MIN)
        omega = OMEGA_MIN
    return omega
