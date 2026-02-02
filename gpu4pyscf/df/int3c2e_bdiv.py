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
import warnings
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf import gto
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, conc_env
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, dist_matrix, asarray, hermi_triu, transpose_sum,
    ndarray, get_avail_mem)
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD, SortedMole, RysIntEnvVars
from gpu4pyscf.gto.mole import basis_seg_contraction, extract_pgto_params, cart2sph_by_l
from gpu4pyscf.scf.jk import (
    g_pair_idx, _nearest_power2, _scale_sp_ctr_coeff, _create_q_cond,
    SHM_SIZE, libvhf_rys)
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'aux_e2',
]

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 54
THREADS = 256
POOL_SIZE = 25600

def aux_e2(mol, auxmol):
    r'''
    3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator(
        reorder_aux=True, cart=mol.cart)[:2]
    aux_coef = int3c2e_opt.aux_coeff
    aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
    aux_coef[aux_sorting] = tmp
    j3c = eval_j3c()
    j3c = j3c.dot(aux_coef)

    nao = mol.nao
    naux = auxmol.nao
    pair_address = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)[0]
    rows, cols = divmod(pair_address, nao)
    out = cp.zeros((nao, nao, naux))
    out[cols,rows] = j3c
    out[rows,cols] = j3c
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
    eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator(
        reorder_aux=True, cart=mol.cart)[:2]
    aux_coef = int3c2e_opt.auxmol.ctr_coeff
    aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
    aux_coef[aux_sorting] = tmp
    j3c = eval_j3c()
    j3c = j3c.dot(aux_coef)
    pair_address = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)[0]
    rows, cols = divmod(pair_address, mol.nao)
    return j3c, rows, cols

def contract_int3c2e_dm(mol, auxmol, dm):
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    dm = int3c2e_opt.mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm)
    return int3c2e_opt.auxmol.apply_CT_dot(auxvec, axis=-1)

def contract_int3c2e_auxvec(mol, auxmol, auxvec):
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    auxvec = int3c2e_opt.auxmol.C_dot_mat(auxvec)
    vj = int3c2e_opt.contract_auxvec(auxvec)
    return int3c2e_opt.mol.apply_CT_mat_C(vj)

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = SortedMole.from_mol(
            mol, allow_replica=True, allow_split_seg_contraction=False)
        self.auxmol = SortedMole.from_mol(auxmol)
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

    def int3c2e_evaluator(self, ao_pair_batch_size=None, aux_batch_size=None,
                          reorder_aux=False, cart=None,
                          omega=None, lr_factor=None, sr_factor=None):
        if self._int3c2e_envs is None:
            self.build()
        mol = self.mol
        auxmol = self.auxmol
        omega, lr_factor, sr_factor = _check_rsh(mol, omega, lr_factor, sr_factor)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(omega, gout_width=54)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block[0]*4)
        if cart is None:
            cart = mol.mol.cart
        ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], self.bas_ij_cache, cart)

        if ao_pair_batch_size is None:
            pair_splits = [0, len(shl_pair_offsets)-1]
            ao_pair_offsets = [0, ao_pair_loc[-1].get()]
        else:
            ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
            pair_splits = splits_by_blocksize(ao_pair_offsets, ao_pair_batch_size)
            ao_pair_offsets = ao_pair_offsets[pair_splits]

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
        uniq_l_ctr_aux = auxmol.uniq_l_ctr
        aux_loc = auxmol.ao_loc
        if aux_batch_size is None:
            ksh_offsets_cpu = l_ctr_aux_offsets
            aux_splits = [0, len(ksh_offsets_cpu)-1]
        else:
            l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
                l_ctr_aux_offsets, uniq_l_ctr_aux, aux_batch_size)
            ksh_offsets_cpu = l_ctr_aux_offsets
            aux_splits = range(len(ksh_offsets_cpu))
        aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
        if reorder_aux:
            aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
        else:
            aux_sorting = slice(aux_loc[-1])

        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
        shl_pair_batches = len(ao_pair_offsets) - 1
        aux_batches = len(aux_offsets) - 1
        logger.debug1(self.mol, 'sp_batches = %d, ksh_batches = %d',
                      shl_pair_batches, aux_batches)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, POOL_SIZE))
        kern = libvhf_rys.fill_int3c2e
        int3c2e_envs = self.int3c2e_envs

        def evaluate_j3c(shl_pair_batch_id=0, aux_batch_id=0, out=None):
            pair_split0 = pair_splits[shl_pair_batch_id]
            pair_split1 = pair_splits[shl_pair_batch_id+1]
            ao_pair_offset = ao_pair_offsets[shl_pair_batch_id]
            nao_pair = ao_pair_offsets[shl_pair_batch_id+1] - ao_pair_offset

            aux_split0 = aux_splits[aux_batch_id]
            aux_split1 = aux_splits[aux_batch_id+1]
            ksh0 = ksh_offsets_cpu[aux_split0]
            ksh1 = ksh_offsets_cpu[aux_split1]
            aux_ao_offset = aux_loc[ksh0]
            naux = aux_loc[ksh1] - aux_ao_offset
            out = ndarray((nao_pair, naux), buffer=out)
            if not cart:
                out[:] = 0.
            if out.size == 0:
                return out
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.c_double(omega),
                ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(pair_split1 - pair_split0),
                ctypes.c_int(aux_split1 - aux_split0),
                ctypes.cast(shl_pair_offsets[pair_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ao_pair_offset), ctypes.c_int(aux_ao_offset),
                ctypes.c_int(naux), ctypes.c_int(reorder_aux),
                ctypes.c_int(not cart))
            if err != 0:
                raise RuntimeError('fill_int3c2e kernel failed')
            return out
        return evaluate_j3c, aux_sorting, ao_pair_offsets, aux_offsets

    def int3c2e_bdiv_generator(self, cutoff=1e-14, batch_size=None, verbose=None):
        '''An iterator to generate eri3c blocks using the block-divergent
        integral parallelism
        '''
        warnings.warn('int3c2e_bdiv_generator is deprecated')
        evaluate_j3c, _, _, aux_offsets = self.int3c2e_evaluator(
            aux_batch_size=batch_size, reorder_aux=False)
        aux_batches = len(aux_offsets) - 1
        for batch_id in range(aux_batches):
            yield evaluate_j3c(aux_batch_id=batch_id)

    def create_ao_pair_mapping(self):
        warnings.warn('create_ao_pair_mapping is deprecated')
        return self.pair_and_diag_indices(original_ao_order=True)[0]

    @property
    def coeff(self):
        return self.mol.ctr_coeff

    @property
    def aux_coeff(self):
        return self.auxmol.ctr_coeff

    def contract_dm(self, dm, hermi=0):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        assert dm.shape[-1] == mol.nao
        assert dm.dtype == np.float64
        assert dm.flags.c_contiguous
        nbas_aux = auxmol.nbas
        if hermi != 1:
            dm = transpose_sum(dm, inplace=False)

        dm_ndim = dm.ndim
        if dm_ndim == 2:
            dm = dm[None]
        n_dm = len(dm)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block[0]*16)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        int3c2e_envs = self.int3c2e_envs
        naux = auxmol.nao
        vj_aux = cp.zeros((n_dm, naux))
        err = libvhf_rys.contract_int3c2e_dm(
            ctypes.cast(vj_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(naux),
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
        if dm_ndim == 2:
            vj_aux = vj_aux[0]
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

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, gout_width=30)
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

    def orbital_pair_cart2sph(self, compressed_eri3c):
        '''Transforms the AO of the compressed eri3c from Cartesian to spherical basis'''
        mol = self.mol
        uniq_l = mol.uniq_l_ctr[:,0]
        bas_ij_idx = mol.aggregate_shl_pairs(self.bas_ij_cache, 1000000000)[0]
        cart_pair_loc = get_ao_pair_loc(uniq_l, self.bas_ij_cache, cart=True)
        assert compressed_eri3c.shape[0] == cart_pair_loc[-1].get(), \
                'compressed_eri3c might be already transformed into spherical GTOs'

        sph_pair_loc = get_ao_pair_loc(uniq_l, self.bas_ij_cache, cart=False)
        nao_pair = sph_pair_loc[-1].get()
        naux = compressed_eri3c.shape[1]
        out = cp.zeros((nao_pair, naux))
        int3c2e_envs = self.int3c2e_envs
        libvhf_rys.int3c2e_cart2sph(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed_eri3c.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(sph_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(cart_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_int(naux), ctypes.c_int(mol.nbas))
        return out

    def pair_and_diag_indices(self, cart=None, original_ao_order=True):
        '''
        original_ao_order:
            controls whether to produce pair addresses corresponding to the
            original Mole (without sorting basis).
        '''
        mol = self.mol
        if cart is None:
            cart = mol.mol.cart
        nbas = mol.nbas
        ao_loc = mol.ao_loc_nr(cart=cart)
        nao = ao_loc[-1]
        if original_ao_order:
            dims = ao_loc[1:] - ao_loc[:-1]
            dims, tmp = np.empty_like(dims), dims
            dims[mol.sorted_idx] = tmp
            ao_loc = cp.asarray(np.append(0, np.cumsum(dims)))
            sorted_idx = cp.asarray(mol.sorted_idx)

        ao_loc = cp.asarray(ao_loc)
        uniq_l = mol.uniq_l_ctr[:,0]
        if cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
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
            if original_ao_order:
                ish = sorted_idx[ish]
                jsh = sorted_idx[jsh]
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

def int3c2e_scheme(omega=0, gout_width=None, shm_size=SHM_SIZE):
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
    conventional order [s,s,...,px,py,pz,px,py,pz,pz,...]. This function returns
    aux_sorting which maps the addresses of the two storge formats.
    Specifically, array_sss_pxpypz_pxpypz = array_sss_pxpx_pypy_pzpz[aux_sorting]
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

def get_ao_pair_loc(uniq_l, bas_ij_cache, cart=True):
    '''
    For each primitive shell-pair in bas_ij_idx, ao_pair_loc points to the
    addresses of first element for the contracted pair-GTOs. In each
    shell-pair, there are nfij elements. Note, the nfij elements are
    sorted as [nfj,nfi] (in F-order).
    '''
    if cart:
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
    else:
        nf = uniq_l * 2 + 1
    ao_pair_loc = []
    p0 = p1 = 0
    for (i, j), bas_ij in bas_ij_cache.items():
        nfij = nf[i] * nf[j]
        p0, p1 = p1, p1 + nfij * len(bas_ij)
        ao_pair_loc.append(cp.arange(p0, p1, nfij, dtype=np.int32))
    ao_pair_loc.append(np.int32(p1))
    ao_pair_loc = cp.asarray(cp.hstack(ao_pair_loc), dtype=np.int32)
    return ao_pair_loc

def int2c2e(mol):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e
    return int2c2e(mol)

def int2c2e_ip1(mol):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1
    return int2c2e_ip1(mol)

def _check_rsh(mol, omega, lr_factor, sr_factor):
    '''
    The parameters for exchange part of the range-separation hybrid functional:
    lr_factor * erf(|omega|r12)/r12 + sr_factor * erfc(|omega|r12)/r12
    '''
    if omega is None:
        omega = mol.omega
    elif sr_factor is not None:
        omega = -abs(omega)
    elif lr_factor is not None:
        omega = abs(omega)

    if omega < 0: # short-range Coulomb
        if sr_factor is None:
            sr_factor = 1
        if lr_factor is None:
            lr_factor = 0
    else: # long-range or full-range Coulomb
        if sr_factor is None:
            sr_factor = 0
        if lr_factor is None:
            lr_factor = 1
    return omega, lr_factor, sr_factor
