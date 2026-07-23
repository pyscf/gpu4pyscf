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

from functools import lru_cache
import ctypes
import math
import numpy as np
import cupy as cp
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_EXP, PTR_COORD, conc_env
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, transpose_sum
from gpu4pyscf.gto.mole import SortedGTO, PTR_BAS_COORD, RysIntEnvVars
from gpu4pyscf.scf.jk import _scale_sp_ctr_coeff, _nearest_power2, SHM_SIZE
from gpu4pyscf.df.int3c2e_bdiv import _conc_locs, LMAX, L_AUX_MAX, THREADS
from gpu4pyscf.scf.j_engine import (
    libvhf_md, _cache_q_cond_and_non0pairs, _dm_to_Rt, _Rt_to_dm)

def contract_int3c2e_dm(mol, auxmol, dm, hermi=0, int3c2e_opt=None):
    if int3c2e_opt is None:
        int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    dm = int3c2e_opt.mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, hermi)
    return int3c2e_opt.auxmol.apply_CT_dot(auxvec, axis=-1)

def contract_int3c2e_auxvec(mol, auxmol, auxvec, int3c2e_opt=None):
    if int3c2e_opt is None:
        int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    auxvec = int3c2e_opt.auxmol.apply_C_dot(auxvec, axis=-1)
    vj = int3c2e_opt.contract_auxvec(auxvec)
    return int3c2e_opt.mol.apply_CT_mat_C(vj)

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        self.auxmol = auxmol
        self.int3c2e_envs = None

    def build(self, cutoff=1e-12):
        log = logger.new_logger(self.mol)
        cput0 = log.init_timer()
        mol = self.mol = SortedGTO.from_mol(
            self.mol, decontract=True, diffuse_cutoff=1e200)
        # very high angular momentum basis are processed on CPU
        lmax = mol.uniq_l_ctr[:,0].max()
        assert lmax <= LMAX

        _atm = cp.array(mol._atm)
        _bas = cp.array(mol._bas)
        _env = cp.array(_scale_sp_ctr_coeff(mol))
        ao_loc = cp.array(mol.ao_loc)
        rys_envs = RysIntEnvVars.new(mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)
        self.bas_pair_cache = _cache_q_cond_and_non0pairs(mol, rys_envs, cutoff)
        log.timer('Initialize q_cond', *cput0)

        auxmol = self.auxmol = SortedGTO.from_mol(
            self.auxmol, decontract=True, diffuse_cutoff=1e200)

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[mol.nbas:,PTR_BAS_COORD] += off
        self._atm = _atm_cpu
        self._bas = _bas_cpu
        self._env = _env_cpu

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(mol.ao_loc, auxmol.ao_loc_nr(cart=True))
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)
        self.int3c2e_envs = RysIntEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)

        shl_pair_idx = [x[0].get() for x in self.bas_pair_cache.values()]
        # the bas_ij_idx offset for each blockIdx.x
        self.shl_pair_offsets = np.cumsum(
            [0] + [len(x) for x in shl_pair_idx], dtype=np.int32)
        self.shl_pair_idx = shl_pair_idx = np.hstack(shl_pair_idx)
        ls = np.asarray(mol._bas[:,ANG_OF], dtype=np.int32)
        ll = ls[:,None] + ls
        ll = ll.ravel()[shl_pair_idx]
        xyz_size = (ll+1)*(ll+2)*(ll+3)//6
        self.pair_loc = np.cumsum(np.append(np.int32(0), xyz_size.ravel()), dtype=np.int32)
        return self

    def contract_dm(self, dm, hermi=0):
        if self.int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        ao_loc = mol.ao_loc
        nao = ao_loc[-1]
        naux = auxmol.nao_nr(cart=True)
        assert dm.shape[-1] == nao, 'Requires transforming dm: mol.apply_C_mat_CT(dm)'

        dm_ndim = dm.ndim
        if dm_ndim == 2:
            dm = dm[None]
        n_dm = len(dm)
        dm = cp.asarray(dm.reshape(n_dm,nao,nao), order='C')
        if hermi != 1:
            dm = transpose_sum(dm)

        nsp_lookup, shm_size =  _int3c2e_dm_scheme()
        shl_pair_idx = asarray(self.shl_pair_idx, dtype=np.int32)
        pair_ij_offsets = cp.asarray(self.shl_pair_offsets, dtype=np.int32)
        sp_blocks = len(pair_ij_offsets) - 1
        log.debug1('sp_blocks = %d, shm_size = %d B', sp_blocks, shm_size)

        dm_xyz_size = int(self.pair_loc[-1])
        pair_loc = cp.asarray(self.pair_loc, dtype=np.int32)
        dm_xyz = cp.empty(dm_xyz_size)
        omega = mol.omega
        int3c2e_envs = self.int3c2e_envs
        vj_aux = cp.zeros((n_dm, naux))
        for i_dm in range(n_dm):
            _dm_to_Rt(mol, dm[i_dm], shl_pair_idx, pair_loc, int3c2e_envs, out=dm_xyz)
            err = libvhf_md.contract_int3c2e_dm(
                ctypes.cast(vj_aux[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.c_int(naux),
                ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size),
                ctypes.c_int(sp_blocks),
                ctypes.c_int(auxmol.nbas),
                ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError('contract_int3c2e_dm kernel failed')
        if hermi == 1:
            vj_aux *= 2
        if dm_ndim == 2:
            vj_aux = vj_aux[0]
        log.timer_debug1('processing contract_int3c2e_dm', *t0)
        return vj_aux

    def contract_auxvec(self, auxvec):
        if self.int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        aux_loc = auxmol.ao_loc
        naux = aux_loc[-1]
        assert auxvec.shape[-1] == naux

        auxvec_ndim = auxvec.ndim
        auxvec = cp.asarray(auxvec.reshape(-1,naux), order='C')
        n_dm = len(auxvec)

        l = auxmol._bas[:,ANG_OF]
        nf3 = (l+1)*(l+2)*(l+3)//6
        aux_xyz_loc = np.asarray(np.append(0, np.cumsum(nf3)), dtype=np.int32)

        nsp_lookup, shm_size = _int3c2e_auxvec_scheme()
        shl_pair_idx = asarray(self.shl_pair_idx, dtype=np.int32)
        pair_ij_offsets = asarray(self.shl_pair_offsets, dtype=np.int32)
        sp_blocks = len(pair_ij_offsets) - 1

        # Split auxbasis into small batches for load balance
        ksh_offsets = []
        k0 = k1 = mol.nbas
        for n in auxmol.l_ctr_counts:
            k0, k1 = k1, k1 + n
            ksh_offsets.append(np.arange(k0, k1, 16, dtype=np.int32))
        ksh_offsets.append(np.int32(k1))
        ksh_offsets = asarray(np.hstack(ksh_offsets, dtype=np.int32))
        ksh_blocks = len(ksh_offsets) - 1

        log.debug1('sp_blocks = %d, ksh_blocks = %d, shm_size = %d B',
                   sp_blocks, ksh_blocks, shm_size)

        nao = mol.nao
        omega = mol.omega
        aux_xyz = cp.empty(aux_xyz_loc[-1])
        vj_xyz_size = self.pair_loc[-1]
        vj_xyz = cp.zeros((n_dm, vj_xyz_size))
        vj = cp.empty((n_dm, nao, nao))
        int3c2e_envs = self.int3c2e_envs
        aux_loc = asarray(aux_loc)
        aux_xyz_loc = asarray(aux_xyz_loc)
        pair_loc = asarray(self.pair_loc)
        for i_dm in range(n_dm):
            libvhf_md.aux_to_Rt(
                ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
                ctypes.cast(auxvec[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(aux_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_xyz_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(auxmol.nbas))
            err = libvhf_md.contract_int3c2e_auxvec(
                ctypes.cast(vj_xyz[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.c_int(naux),
                ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size),
                ctypes.c_int(sp_blocks),
                ctypes.c_int(ksh_blocks),
                ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_xyz_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError('contract_int3c2e_auxvec kernel failed')
            _Rt_to_dm(mol, vj_xyz[i_dm], shl_pair_idx, pair_loc, int3c2e_envs, out=vj[i_dm])
        log.timer_debug1('processing contract_int3c2e_auxvec', *t0)

        if auxvec_ndim == 1:
            vj = vj[0]
        return vj

@lru_cache
def _int3c2e_dm_scheme():
    li = np.arange(LMAX*2+1)
    lk = np.arange(L_AUX_MAX+1)
    order = li[:,None] + lk
    nf3ijkl = (order + 1) * (order + 2) * (order + 3) // 6
    Rt_swap_size = np.array([42, 42, 42, 42, 42, 42, 30])
    Rt_stride = (nf3ijkl + Rt_swap_size-1) // Rt_swap_size
    nsp_max = THREADS // Rt_stride

    unit = order+1 + nf3ijkl
    nsp_per_block = SHM_SIZE //(unit*8)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    nsp_per_block = _nearest_power2(nsp_per_block)
    nsp_per_block[nsp_per_block>THREADS] = THREADS

    shm_size = (nsp_per_block * unit).max() * 8
    nsp_lookup = cp.asarray(nsp_per_block, dtype=np.int32)
    return nsp_lookup, shm_size

@lru_cache
def _int3c2e_auxvec_scheme():
    li = np.arange(LMAX*2+1)
    lk = np.arange(L_AUX_MAX+1)
    nf3k = (L_AUX_MAX+1)*(L_AUX_MAX+2)*(L_AUX_MAX+3)//6
    order = li[:,None] + lk
    nf3ijkl = (order + 1) * (order + 2) * (order + 3) // 6
    Rt_swap_size = np.array([35, 35, 35, 35, 35, 21, 21])
    Rt_stride = (nf3ijkl + Rt_swap_size-1) // Rt_swap_size
    nfij = (li + 1) * (li + 2) * (li + 3) // 6
    IJ_SIZE = np.array([35, 21, 15, 11, 8, 8, 8])
    Rt_stride_min = (nfij[:,None] + IJ_SIZE-1) // IJ_SIZE
    Rt_stride = np.where(Rt_stride > Rt_stride_min, Rt_stride, Rt_stride_min)

    nsp_max = THREADS // Rt_stride
    unit = order+1 + nf3ijkl #+ order * (order + 1) * (order + 2) // 6
    shm_size = SHM_SIZE - nf3k * 8
    nsp_per_block = shm_size //(unit*8)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    nsp_per_block = _nearest_power2(nsp_per_block)
    nsp_per_block[nsp_per_block>THREADS] = THREADS

    shm_size = ((nsp_per_block * unit).max() + nf3k) * 8
    nsp_lookup = asarray(nsp_per_block, dtype=np.int32)
    return nsp_lookup, shm_size
