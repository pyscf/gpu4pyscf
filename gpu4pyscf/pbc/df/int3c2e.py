# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
3-center 2-electron integrals with perodicity
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, conc_env
from pyscf.pbc import tools as pbctools
from pyscf.pbc.gto.cell import _extract_pgto_params
from pyscf.pbc.df.rsdf_builder import estimate_rcut
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.pbc.df.ft_ao import libpbc, init_constant

__all__ = [
    'aux_e2',
]

libpbc.fill_int3c2e.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 45
THREADS = 256

def sr_aux_e2(cell, auxcell, omega, kpts=None, bvk_kmesh=None):
    r'''
    Short-range 3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    kern = Int3c2eOpt(cell, auxcell, omega, kpts, bvk_kmesh).gen_int3c2e_kernel()
    return kern()

def create_img_idx(cell, bvkcell, Ls, int3c2e_envs):
    '''integral screening'''
    # consider only the most diffused component of a basis
    exps, cs = _extract_pgto_params(cell, 'min')
    ls = cell._bas[:,ANG_OF]
    exps = cp.asarray(exps, dtype=np.float32)
    log_cs = np.log(np.abs(cs * ((2*ls+1)/(4*np.pi))**.5))
    log_cs = cp.asarray(log_cs, np.float32)
    nbas = cell.nbas
    nk = bvkcell.nbas // nbas
    nimgs = len(Ls)

    def gen_img_idx(ish0, ish1, jsh0, jsh1):
        nish = ish1 - ish0
        njsh = jsh1 - jsh0
        ij_pairs = nk * nish * nk * njsh
        mask = cp.zeros((ij_pairs, nimgs**2), dtype=np.int8)
        img_counts = cp.empty(ij_pairs, dtype=np.int32)
        libpbc.int3c2e_q_mask(
            ctypes.cast(mask.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
            ctypes.cast(exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_cs.data.ptr, ctypes.c_void_p),
            ctypes.c_int(cell.natm))

        img_counts_mask = img_counts > 0
        img_counts = img_counts[img_counts_mask]
        bas_idx = cp.asarray(cp.argsort(img_counts)[::-1], dtype=np.int32, order='C')
        img_offsets = cp.empty(bas_idx.size+1, dtype=np.int32)
        cp.cumsum(img_counts[bas_idx], out=img_offsets[1:])
        img_offsets[0] = 0

        img_idx = cp.empty(int(img_offsets[-1]), dtype=np.int32)
        libpbc.int3c2e_img_idx(
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(mask.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ij_pairs), ctypes.c_int(nimgs))

        #TODO: only tril part when i == j
        K = cp.arange(nk)
        Ki = (K[:,None] * nbas + cp.arange(ish0, ish1)).astype(np.int32)
        Kj = (K[:,None] * nbas + cp.arange(jsh0, jsh1)).astype(np.int32)
        bvk_nbas = nk * nbas
        bas_ij = Ki.reshape(-1,1) * bvk_nbas + Kj.ravel()
        bas_ij = bas_ij.ravel()[img_counts_mask]
        bas_ij = cp.asarray(bas_ij[bas_idx], dtype=np.int32)
        return img_idx, img_offsets, bas_ij
    return gen_img_idx

class Int3c2eOpt:
    def __init__(self, cell, auxcell, omega, kpts=None, bvk_kmesh=None):
        assert omega < 0
        self.cell = cell
        cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
        self.sorted_cell = cell
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.coeff = cp.asarray(coeff, dtype=np.complex128)
        self.sorted_cell.omega = omega

        self.auxcell = auxcell
        cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(auxcell, tile=1)
        self.sorted_auxcell = cell
        self.uniq_l_ctr_aux = uniq_l_ctr
        self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.aux_coeff = cp.asarray(coeff, dtype=np.complex128)
        self.sorted_auxcell.omega = omega

        if kpts is not None and bvk_kmesh is None:
            bvk_kmesh = kpts_to_kmesh(cell, kpts)

        # create BVK super-cell
        if bvk_kmesh is None:
            bvkcell = cell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell
        self.bvk_kmesh = bvk_kmesh
        self.kpts = kpts
        self.omega = omega

    def gen_int3c2e_kernel(self, cutoff=None, verbose=None):
        cell = self.sorted_cell
        auxcell = self.sorted_auxcell
        coeff = self.coeff
        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets
        bvk_kmesh = self.bvk_kmesh
        bvkcell = self.bvkcell
        kpts = self.kpts

        log = logger.new_logger(cell, verbose)
        cput0 = log.init_timer()
        rcut = estimate_rcut(cell, auxcell, self.omega)
        Ls = cp.asarray(bvkcell.get_lattice_Ls(rcut=rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        log.debug('rcut = %g, nimgs = %d', rcut, len(Ls))

        if bvk_kmesh is None:
            bvkmesh_Ls = cp.zeros((1, 3))
        else:
            bvkmesh_Ls = cp.asarray(
                k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True))

        if cutoff is None:
            omega = cell.omega
            aux_exp = np.hstack(auxcell.bas_exps()).min()
            cell_exp = np.hstack(cell.bas_exps()).min()
            if omega == 0:
                theta = 1./(1./cell_exp + 1./aux_exp)
            else:
                theta = 1./(1./cell_exp + 1./aux_exp + omega**-2)
            lattice_sum_factor = max(2*np.pi*cell.rcut/(cell.vol*theta), 1)
            cutoff = cell.precision / lattice_sum_factor * .1
            log.debug1('int3c_kernel integral omega=%g theta=%g cutoff=%g',
                       omega, theta, cutoff)

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            auxcell._atm, auxcell._bas, _scale_sp_ctr_coeff(auxcell))
        bvk_ao_loc = bvkcell.ao_loc
        aux_loc = auxcell.ao_loc
        bvk_nbas = bvkcell.nbas

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(bvk_ao_loc, aux_loc)
        bvk_ncells = len(bvkmesh_Ls)
        nimgs = len(Ls)
        int3c2e_envs = Int3c2eEnvVars(
            bvkcell.natm, bvk_nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
            Ls.data.ptr, math.log(cutoff),
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)

        gen_img_idx = create_img_idx(cell, bvkcell, Ls, int3c2e_envs)

        nbas = cell.nbas
        nao, nao_orig = coeff.shape
        ao_loc = bvkcell.ao_loc
        uniq_l = uniq_l_ctr[:,0]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        init_constant(cell)
        kern = libpbc.fill_int3c2e
        cp.cuda.Stream.null.synchronize()
        log.timer_debug1('initialize int3c2e_kernel', *cput0)

        def int3c2e_kernel(kpts=kpts, j_only=False, shls_slice=None, transform_ao=True):
            '''
            FT tensor is first computed in the basis of sorted_cell, which
            transform_ao requires to transform AOs to their original order
            '''
            t1 = log.init_timer()
            timing_collection = {}
            kern_counts = 0
            if kpts is None:
                kpts = np.zeros((1, 3))
            nkpts = len(kpts)

            if j_only:
                out = np.zeros((nkpts, nao, nao, auxcell.nao), dtype=np.complex128)
            else:
                out = np.zeros((nkpts, nao, nkpts, nao, auxcell.nao), dtype=np.complex128)

            ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
            for i, j in ij_tasks:
                li = uniq_l[i]
                lj = uniq_l[j]
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                nrow = ao_loc[ish1] - ao_loc[ish0]
                ncol = ao_loc[jsh1] - ao_loc[jsh0]
                img_idx, img_offsets, bas_ij_idx = gen_img_idx(ish0, ish1, jsh0, jsh1)

                for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                    ksh0, ksh1 = self.l_ctr_aux_offsets[k:k+2]
                    naux = aux_loc[ksh1] - aux_loc[ksh0]
                    eri3c = cp.zeros((bvk_ncells,nrow, bvk_ncells, ncol,naux))
                    lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                    scheme = int3c2e_scheme(cell, li, lj, lk)
                    log.debug2('int3c2e_scheme for %s: %s', lll, scheme)
                    err = kern(
                        ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
                        ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                        (ctypes.c_int*6)(ish0, ish1, jsh0, jsh1,
                                         bvk_nbas+ksh0, bvk_nbas+ksh1),
                        ctypes.c_int(bas_ij_idx.size),
                        ctypes.c_uint16(nrow), ctypes.c_uint16(ncol),
                        ctypes.c_uint16(naux),
                        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                        _atm_cpu.ctypes, ctypes.c_int(bvkcell.natm),
                        _bas_cpu.ctypes, ctypes.c_int(bvkcell.nbas), _env_cpu.ctypes)
                    if err != 0:
                        raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')

                    if log.verbose >= logger.DEBUG1:
                        t1, t1p = log.timer_debug1(f'processing {lll}', *t1), t1
                        if lll not in timing_collection:
                            timing_collection[lll] = 0
                        timing_collection[lll] += t1[1] - t1p[1]
                        kern_counts += 1

                log.debug1('transform BvK-cell to k-points')
                #TODO:if kptjs is not None:
                #TODO:    kptjs = cp.asarray(kptjs, order='C').reshape(-1,3)
                #TODO:    expLk = cp.exp(1j*cp.dot(bvkmesh_Ls, kptjs.T))
                #TODO:    out = contract('Lk,LpqG->kGpq', expLk, out)
                #TODO:eri[:,i0:i1,:,j0:j1,k0:k1] = out

            if log.verbose >= logger.DEBUG1:
                log.debug1('kernel launches %d', kern_counts)
                for lll, t in timing_collection.items():
                    log.debug1('%s wall time %.2f', lll, t)

            #if transform_ao:
            #    log.debug1('transform basis')
            #    #:out = einsum('pqLG,pi,qj->LGij', out, coeff, coeff)
            #    out = contract('kGpq,qj->kGpj', out, coeff)
            #    out = contract('kGpj,pi->kGij', out, coeff)

            log.timer('int3c2e', *cput0)
            return out
        return int3c2e_kernel

    def int3c2e_kernel_iterator(self, cutoff=None, verbose=None):
        cell = self.sorted_cell
        auxcell = self.sorted_auxcell
        coeff = self.coeff
        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets
        bvk_kmesh = self.bvk_kmesh
        bvkcell = self.bvkcell
        kpts = self.kpts

        log = logger.new_logger(cell, verbose)
        cput0 = log.init_timer()
        rcut = estimate_rcut(cell, auxcell, self.omega)
        Ls = cp.asarray(bvkcell.get_lattice_Ls(rcut=rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug('rcut = %g, nimgs = %d', rcut, nimgs)

        if cutoff is None:
            omega = cell.omega
            aux_exp = np.hstack(auxcell.bas_exps()).min()
            cell_exp = np.hstack(cell.bas_exps()).min()
            if omega == 0:
                theta = 1./(1./cell_exp + 1./aux_exp)
            else:
                theta = 1./(1./cell_exp + 1./aux_exp + omega**-2)
            lattice_sum_factor = max(2*np.pi*cell.rcut/(cell.vol*theta), 1)
            cutoff = cell.precision / lattice_sum_factor * .1
            log.debug1('int3c_kernel integral omega=%g theta=%g cutoff=%g',
                       omega, theta, cutoff)

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            auxcell._atm, auxcell._bas, _scale_sp_ctr_coeff(auxcell))
        bvk_ao_loc = bvkcell.ao_loc
        aux_loc = auxcell.ao_loc
        bvk_nbas = bvkcell.nbas

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(bvk_ao_loc, aux_loc)
        bvk_ncells = bvkcell.nbas // cell.nbas
        int3c2e_envs = Int3c2eEnvVars(
            bvkcell.natm, bvk_nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
            Ls.data.ptr, math.log(cutoff),
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)

        gen_img_idx = create_img_idx(cell, bvkcell, Ls, int3c2e_envs)

        ao_loc = bvkcell.ao_loc
        uniq_l = uniq_l_ctr[:,0]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        init_constant(cell)
        kern = libpbc.fill_int3c2e
        cp.cuda.Stream.null.synchronize()
        log.timer_debug1('initialize int3c2e_kernel', *cput0)

        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            nrow = ao_loc[ish1] - ao_loc[ish0]
            ncol = ao_loc[jsh1] - ao_loc[jsh0]
            img_idx, img_offsets, bas_ij_idx = gen_img_idx(ish0, ish1, jsh0, jsh1)

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = self.l_ctr_aux_offsets[k:k+2]
                naux = aux_loc[ksh1] - aux_loc[ksh0]
                eri3c = cp.zeros((bvk_ncells,nrow, bvk_ncells, ncol,naux))
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(cell, li, lj, lk)
                log.debug2('int3c2e_scheme for %s: %s', lll, scheme)
                err = kern(
                    ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(ish0, ish1, jsh0, jsh1,
                                     bvk_nbas+ksh0, bvk_nbas+ksh1),
                    ctypes.c_int(bas_ij_idx.size),
                    ctypes.c_uint16(nrow), ctypes.c_uint16(ncol),
                    ctypes.c_uint16(naux),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(bvkcell.natm),
                    _bas_cpu.ctypes, ctypes.c_int(bvkcell.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {lll}', *t1), t1
                    if lll not in timing_collection:
                        timing_collection[lll] = 0
                    timing_collection[lll] += t1[1] - t1p[1]
                    kern_counts += 1
                yield eri3c

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for lll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', lll, t)

    def fill_int3c2e(self, cutoff=None, verbose=None):
        raise NotImplementedError

    def contract_k_kernel(self, kpts=kpts, transform_ao=True):
        raise NotImplementedError

    def contract_j_kernel(self, kpts=kpts, transform_ao=True):
        raise NotImplementedError

    def contract_j_aux_kernel(self, kpts=kpts, transform_ao=True):
        raise NotImplementedError

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('bvk_ncells', ctypes.c_uint16),
        ('nimgs', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('img_coords', ctypes.c_void_p),
        ('log_cutoff', ctypes.c_float),
    ]

def _conc_locs(ao_loc1, ao_loc2):
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return cp.array(comp_loc, dtype=np.int32)

def int3c2e_scheme(cell, li, lj, lk, shm_size=SHM_SIZE):
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    gout_size = nfi * nfj * nfk
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    # Align nksh*gout_stride to warp size
    nksh_min = 32 // gout_stride

    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3
    nksh_nsp_max = shm_size//(unit*8)
    nksh_nsp_max = _nearest_power2(nksh_nsp_max)
    if nksh_nsp_max < nksh_min:
        raise RuntimeError('GOUT_WIDTH too small or not enough shared memory')

    nksh_per_block = nksh_min
    nsp_per_block = THREADS // (nksh_per_block*gout_stride)
    return nksh_per_block, gout_stride, nsp_per_block
