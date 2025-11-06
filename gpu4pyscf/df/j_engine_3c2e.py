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
from pyscf import lib
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_EXP, PTR_COORD, conc_env
from pyscf.scf import _vhf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, transpose_sum
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import (
    apply_coeff_C_mat_CT, _scale_sp_ctr_coeff, _nearest_power2, SHM_SIZE)
from gpu4pyscf.gto.mole import basis_seg_contraction, cart2sph_by_l
from gpu4pyscf.df.int3c2e_bdiv import (
    Int3c2eEnvVars, _conc_locs, LMAX, L_AUX_MAX, THREADS)
from gpu4pyscf.scf.j_engine import libvhf_md, _to_primitive_bas, _estimate_q_cond

libvhf_md.MD_int3c2e_init(SHM_SIZE)

def contract_int3c2e_dm(mol, auxmol, dm):
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    return int3c2e_opt.contract_dm(dm)

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        self.auxmol = auxmol
        self.sorted_mol = None

    def build(self, cutoff=1e-14):
        mol = self.mol
        log = logger.new_logger(mol)
        cput0 = log.init_timer()
        sorted_mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = \
                group_basis(mol, 1, sparse_coeff=True)
        self.sorted_mol = sorted_mol
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = l_ctr_pad_counts
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        # very high angular momentum basis are processed on CPU
        lmax = uniq_l_ctr[:,0].max()
        assert lmax <= LMAX

        prim_mol, self.prim_to_ctr_mapping = _to_primitive_bas(sorted_mol)
        self.prim_mol = prim_mol

        nbas = prim_mol.nbas
        ao_loc = prim_mol.ao_loc
        if 1:
            q_cond = _estimate_q_cond(prim_mol).get()
        else:
            q_cond = np.empty((nbas,nbas))
            intor = prim_mol._add_suffix('int2e')
            with prim_mol.with_integral_screen(1e-26):
                _vhf.libcvhf.CVHFnr_int2e_q_cond(
                    getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
                    q_cond.ctypes, ao_loc.ctypes,
                    prim_mol._atm.ctypes, ctypes.c_int(prim_mol.natm),
                    prim_mol._bas.ctypes, ctypes.c_int(prim_mol.nbas),
                    prim_mol._env.ctypes)
            q_cond = np.log(q_cond + 1e-300).astype(np.float32)
        log.timer('Initialize q_cond', *cput0)

        auxmol = self.auxmol
        auxmol, aux_idx = group_basis(self.auxmol, tile=1, sparse_coeff=True)[:2]
        self.sorted_auxmol = auxmol
        self.aux_idx = aux_idx

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            prim_mol._atm, prim_mol._bas, _scale_sp_ctr_coeff(prim_mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[prim_mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[prim_mol.nbas:,PTR_BAS_COORD] += off
        self._atm = _atm_cpu
        self._bas = _bas_cpu
        self._env = _env_cpu

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = cp.asarray(_conc_locs(ao_loc, auxmol.ao_loc_nr(cart=True)), dtype=np.int32)
        log_cutoff = math.log(cutoff)
        self.int3c2e_envs = Int3c2eEnvVars.new(
            prim_mol.natm, prim_mol.nbas, _atm, _bas, _env, ao_loc, log_cutoff)

        l_counts = np.bincount(prim_mol._bas[:,ANG_OF])[:LMAX+1]
        n_groups = len(l_counts)
        bas_offsets = np.cumsum(np.append(0, l_counts))
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        # The effective shell pair = ish*nbas+jsh
        shl_pair_idx = []
        nbas = prim_mol.nbas
        for i, j in ij_tasks:
            ish0, ish1 = bas_offsets[i], bas_offsets[i+1]
            jsh0, jsh1 = bas_offsets[j], bas_offsets[j+1]
            mask = q_cond[ish0:ish1,jsh0:jsh1] > log_cutoff
            if i == j:
                mask = np.tril(mask)
            t_ij = (np.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    np.arange(jsh0, jsh1, dtype=np.int32))
            pair_idx = t_ij[mask]
            if pair_idx.size > 0:
                shl_pair_idx.append(pair_idx)

        # the bas_ij_idx offset for each blockIdx.x
        self.shl_pair_offsets = np.cumsum(
            [0] + [x.size for x in shl_pair_idx], dtype=np.int32)
        self.shl_pair_idx = shl_pair_idx = np.hstack(shl_pair_idx)
        ls = np.asarray(prim_mol._bas[:,ANG_OF], dtype=np.int32)
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
        _atm_cpu = self._atm
        _bas_cpu = self._bas
        _env_cpu = self._env
        sorted_mol = self.sorted_mol
        ao_loc = sorted_mol.ao_loc
        naux = self.sorted_auxmol.nao_nr(cart=True)
        prim_mol = self.prim_mol

        nsp_lookup = np.empty([LMAX*2+1,L_AUX_MAX+1], dtype=np.int32)
        lmax = self.uniq_l_ctr[:,0].max()
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
        dm = apply_coeff_C_mat_CT(dm, self.mol, sorted_mol, self.uniq_l_ctr,
                                  self.l_ctr_offsets, self.ao_idx)
        dm = transpose_sum(dm)
        dm_cpu = dm.get()
        dm = None
        dm_xyz_size = self.pair_loc[-1]
        dm_xyz = np.zeros((n_dm, dm_xyz_size))
        _env = _scale_sp_ctr_coeff(prim_mol)
        libvhf_md.Et_dot_dm(
            dm_xyz.ctypes, dm_cpu.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, self.pair_loc.ctypes,
            shl_pair_idx_cpu.ctypes, ctypes.c_int(len(shl_pair_idx_cpu)),
            self.prim_to_ctr_mapping.ctypes,
            ctypes.c_int(prim_mol.nbas), ctypes.c_int(sorted_mol.nbas),
            prim_mol._bas.ctypes, _env.ctypes)
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
            _atm_cpu.ctypes, ctypes.c_int(prim_mol.natm),
            _bas_cpu.ctypes, ctypes.c_int(prim_mol.nbas), _env_cpu.ctypes)
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm kernel failed')
        if log.verbose >= logger.DEBUG1:
            log.timer_debug1('processing contract_int3c2e_dm', *t0)

        if not self.auxmol.cart:
            vj_aux = _vector_cart2sph(self.sorted_auxmol, vj_aux)
        vj_aux[self.aux_idx] = vj_aux
        return vj_aux

def _vector_cart2sph(auxmol, auxvec):
    aux_ls = auxmol._bas[:,ANG_OF]
    lmax = aux_ls.max()
    aux_loc_cart = auxmol.ao_loc_nr(cart=True)[:-1]
    aux_loc_sph = auxmol.ao_loc_nr(cart=False)
    naux_sph = aux_loc_sph[-1]
    aux_loc_sph = aux_loc_sph[:-1]
    out = cp.empty(naux_sph)
    for l in range(lmax+1):
        nf = (l + 1) * (l + 2) // 2
        addrs_for_cart = aux_loc_cart[aux_ls == l]
        addrs_for_sph = aux_loc_sph[aux_ls == l]
        subvec = auxvec[addrs_for_cart[:,None] + np.arange(nf)]
        subvec = subvec.dot(cart2sph_by_l(l))
        out[addrs_for_sph[:,None] + np.arange(l*2+1)] = subvec
    return out
