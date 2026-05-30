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
from gpu4pyscf.scf.j_engine import libvhf_md, _cache_q_cond_and_non0pairs

def contract_int3c2e_dm(mol, auxmol, dm):
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    return int3c2e_opt.contract_dm(dm)

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        self.auxmol = auxmol
        self.sorted_mol = None

    def build(self, cutoff=1e-12):
        mol = self.mol
        log = logger.new_logger(mol)
        cput0 = log.init_timer()
        mol = self.sorted_mol = SortedGTO.from_mol(
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

        auxmol = self.sorted_auxmol = SortedGTO.from_mol(
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

    def contract_dm(self, dm):
        if self.sorted_mol is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        int3c2e_envs = self.int3c2e_envs
        mol = self.sorted_mol
        ao_loc = mol.ao_loc
        naux = self.sorted_auxmol.nao_nr(cart=True)

        nsp_lookup = np.empty([LMAX*2+1,L_AUX_MAX+1], dtype=np.int32)
        lmax = mol.uniq_l_ctr[:,0].max()
        lmax_aux = self.sorted_auxmol._bas[:,ANG_OF].max()
        shm_size = 0
        for lk in range(lmax_aux+1):
            for li in range(lmax*2+1):
                order = li + lk
                nf3k = (lk + 1) * (lk + 2) * (lk + 3) // 6
                nf3ijkl = (order + 1) * (order + 2) * (order + 3) // 6
                unit = order+1 + nf3ijkl + nf3ijkl*order//(order+3)
                nsp_per_block = (SHM_SIZE - nf3k*8) //(unit*8)
                nsp_per_block = min(THREADS, _nearest_power2(nsp_per_block))
                nsp_lookup[li,lk] = nsp_per_block
                shm_size = max(shm_size, nsp_per_block * unit + nf3k)
        shm_size *= 8 # doubles
        nsp_lookup = cp.asarray(nsp_lookup, dtype=np.int32)

        # Adjust the number of shell-pairs in each group for better balance.
        shl_pair_idx_cpu = np.asarray(self.shl_pair_idx, dtype=np.int32)
        shl_pair_idx = asarray(self.shl_pair_idx, dtype=np.int32)
        pair_ij_offsets = cp.asarray(self.shl_pair_offsets, dtype=np.int32)
        sp_blocks = len(pair_ij_offsets) - 1
        log.debug1('sp_blocks = %d, shm_size = %d B', sp_blocks, shm_size)

        dm = cp.asarray(dm)
        assert dm.ndim == 2
        n_dm = 1
        dm = mol.apply_C_mat_CT(dm)
        dm = transpose_sum(dm)
        dm_cpu = dm.get()
        dm = None
        dm_xyz_size = self.pair_loc[-1]
        dm_xyz = np.zeros((n_dm, dm_xyz_size))
        _env = _scale_sp_ctr_coeff(mol)
        libvhf_md.Et_dot_dm(
            dm_xyz.ctypes, dm_cpu.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, self.pair_loc.ctypes,
            shl_pair_idx_cpu.ctypes, ctypes.c_int(len(shl_pair_idx_cpu)),
            ctypes.c_int(mol.nbas),
            mol._bas.ctypes, _env.ctypes)
        dm_xyz = asarray(dm_xyz)
        pair_loc = asarray(self.pair_loc)

        vj_aux = cp.zeros(naux)
        err = libvhf_md.contract_int3c2e_dm(
            ctypes.cast(vj_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(naux),
            ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size),
            ctypes.c_int(sp_blocks),
            ctypes.c_int(self.sorted_auxmol.nbas),
            ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
            ctypes.c_double(mol.omega))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm kernel failed')
        if log.verbose >= logger.DEBUG1:
            log.timer_debug1('processing contract_int3c2e_dm', *t0)

        if not self.auxmol.cart:
            vj_aux = self.sorted_auxmol.apply_CT_dot(vj_aux)
        return vj_aux
