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
Periodic 2-center 2-electron short-range Coulomb integral helper functions
'''

import ctypes
import numpy as np
import cupy as cp
from pyscf.gto import ATOM_OF, PTR_COORD, Mole
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, hermi_triu
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, PBCIntEnvVars, _scale_sp_ctr_coeff)
from gpu4pyscf.df.int3c2e_bdiv import (
    _nearest_power2, SHM_SIZE, L_AUX_MAX, THREADS)
from gpu4pyscf.pbc.df.ft_ao import libpbc, most_diffuse_pgto

__all__ = [
    'int2c2e', 'sr_int2c2e', 'Int2c2eOpt'
]

libpbc.fill_int2c2e.restype = ctypes.c_int

def int2c2e(auxcell, kpts=None, bvk_kmesh=None):
    '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
    if bvk_kmesh is None:
        bvk_kmesh = kpts_to_kmesh(auxcell, kpts, bound_by_supmol=True)
    opt = Int2c2eOpt(auxcell, bvk_kmesh).build()
    return opt.int2c2e(kpts)

def sr_int2c2e(auxcell, omega, kpts=None, bvk_kmesh=None):
    assert omega < 0
    # Adjust the rcut because the default cell.rcut is estimated based on
    # overlap integrals
    rcut = _estimate_sr_2c2e_rcut(auxcell, omega, auxcell.precision*1e-3)
    try:
        auxcell.rcut, rcut_backup = rcut, auxcell.rcut
        auxcell.omega, omega_backup = omega, auxcell.omega
        return int2c2e(auxcell, kpts, bvk_kmesh)
    finally:
        auxcell.rcut = rcut_backup
        auxcell.omega = omega_backup

def int2c2e_ip1_per_atom(auxcell, dm, kpts=None):
    '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
    opt = Int2c2eOpt(auxcell).build()
    return opt.energy_ip1_per_atom(dm, kpts)

def int2c2e_ip1(auxcell, kpts=None, bvk_kmesh=None):
    '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
    if bvk_kmesh is None:
        bvk_kmesh = kpts_to_kmesh(auxcell, kpts, bound_by_supmol=True)
    opt = Int2c2eOpt(auxcell, bvk_kmesh).build()
    return opt.int2c2e_ip1(kpts)

def _estimate_sr_2c2e_rcut(cell, omega, precision=None):
    '''Estimate rcut for SR int2c2e. cell.rcut is likely insufficient to
    converge this integral
    '''
    if precision is None:
        precision = cell.precision
    ak, ck, lk = most_diffuse_pgto(cell)
    theta = 1./(omega**-2 + 2./ak)
    norm_ang = (2*lk+1)/(4*np.pi)
    c1 = ck**2 * norm_ang
    fl = 2
    fac = np.pi**2.5*c1 * theta**(lk*2-.5)
    vol = cell.vol
    rad = vol**(-1./3) * cell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*cell.rcut/(vol*theta) + surface
    fac *= lattice_sum_factor / ak**(lk*2+3) * fl / precision
    rcut = cell.rcut
    rcut = (np.log(fac * rcut**(lk*2-1) + 1.) / theta)**.5
    return rcut

def int2c2e_scheme(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(L_AUX_MAX+1)[:,None]
    lj = np.arange(L_AUX_MAX+1)
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    order = li + lj
    nroots = order//2 + 1
    if omega < 0:
        nroots *= 2 # for short-range
    g_size = (li+1)*(lj+1)
    unit = g_size*3 + nroots*2 + 4
    shm_size = shm_size - (nfi + nfj) * 3 * 4
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        gout_size = nfi * nfj
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    shm_size += (nfi + nfj) * 3 * 4
    return nsp_per_block, gout_stride, shm_size

class Int2c2eOpt:
    def __init__(self, cell, bvk_kmesh=None):
        cell = self.cell = SortedGTO.from_cell(
            cell, allow_replica=True, allow_split_seg_contraction=False)
        assert cell.uniq_l_ctr[:,0].max() <= L_AUX_MAX

        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)

        if isinstance(cell, Mole):
            bvkcell = cell
            bvkmesh_Ls = np.zeros((1, 3))
            Ls = cp.zeros((1, 3))
        else:
            if bvk_ncells == 1:
                bvkcell = cell
                bvkmesh_Ls = np.zeros((1, 3))
            else:
                bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
                # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
                bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
                bvkmesh_Ls = translation_vectors_for_kmesh(cell, bvk_kmesh, True)
            Ls = asarray(bvkcell.get_lattice_Ls(rcut=cell.rcut))
            Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]

        self.bvkcell = bvkcell
        self.bvkmesh_Ls = bvkmesh_Ls
        bvk_ncells = len(bvkmesh_Ls)
        nimgs = len(Ls)
        logger.debug(cell, 'int2c2e_kernel, nimgs = %d', nimgs)
        _env = _scale_sp_ctr_coeff(bvkcell)
        self._rys_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, nimgs,
            bvkcell._atm, bvkcell._bas, _env, bvkcell.ao_loc, Ls)

        self.bas_ij_cache = None

    def build(self):
        cell = self.cell
        bvk_ncells = len(self.bvkmesh_Ls)
        self.bas_ij_cache = bas_ij_cache = {}
        nbas = cell.nbas
        img = cp.arange(bvk_ncells, dtype=np.int32)
        l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
        uniq_l = cell.uniq_l_ctr[:,0]
        ij_tasks = [(i, j) for i in range(len(uniq_l)) for j in range(i+1)]
        for i, j in ij_tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.int32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
            ijsh = ish[:,None] * (nbas*bvk_ncells) + jsh
            if i == j:
                ijsh = ijsh[cp.tril_indices(ish1-ish0)]
            else:
                ijsh = ijsh.ravel()
            idx = (img[:,None] * nbas + ijsh).ravel()
            bas_ij_cache[i, j] = cp.asarray(idx, dtype=np.uint32)
        return self

    def int2c2e(self, kpts=None):
        '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
        from gpu4pyscf.pbc.df.int3c2e import fill_triu_bvk
        if self.bas_ij_cache is None:
            self.build()
        cell = self.cell
        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = len(self.bvkmesh_Ls)

        nsp_per_block, gout_stride, shm_size = int2c2e_scheme(cell.omega, gout_width=60)
        lmax = cell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:lmax+1,:lmax+1].max()
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block*4)

        nbatches_shl_pair = len(shl_pair_offsets) - 1
        rys_envs = self._rys_envs
        nao = cell.nao
        out = cp.empty((bvk_ncells, nao, nao))
        err = libpbc.fill_int2c2e(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('fill_int2c2e failed')

        # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
        out = fill_triu_bvk(out, nao, bvk_kmesh, bvk_axis=0)

        out = cell.apply_CT_mat_C(out)
        if kpts is None:
            out = out[0]
        elif not is_zero(kpts):
            expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(asarray(kpts).T))
            out = contract('lk,lpq->kpq', expLk, out)
        return out

    def int2c2e_ip1(self, kpts=None):
        '''Derivatives of 2c2e Coulomb integrals'''
        assert kpts is None
        if self.bas_ij_cache is None:
            self.build()
        cell = self.cell
        bvk_ncells = len(self.bvkmesh_Ls)
        assert bvk_ncells == 1

        shm_size = SHM_SIZE
        li = np.arange(L_AUX_MAX+1)[:,None]
        lj = np.arange(L_AUX_MAX+1)
        nfi = (li + 1) * (li + 2) // 2
        nfj = (lj + 1) * (lj + 2) // 2
        order = li + lj + 1
        nroots = order//2 + 1
        if cell.omega < 0:
            nroots *= 2 # for short-range
        g_size = (li+2)*(lj+1)
        unit = g_size*3 + nroots*2 + 4
        nsp_max = _nearest_power2(shm_size // (unit*8))
        nsp_per_block = THREADS
        gout_width = 20
        if gout_width is not None:
            gout_size = nfi * nfj
            gout_stride = (gout_size + gout_width-1) // gout_width
            # Round up to the next 2^n
            gout_stride = _nearest_power2(gout_stride, return_leq=False)
            nsp_per_block = THREADS // gout_stride
        nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
        gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
        shm_size = nsp_per_block * (unit*8)
        lmax = cell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:lmax+1,:lmax+1].max()

        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block*4)

        nbatches_shl_pair = len(shl_pair_offsets) - 1
        rys_envs = self._rys_envs
        nao = cell.nao
        out = cp.empty((bvk_ncells*3, nao, nao))
        err = libpbc.fill_int2c2e_ip1(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('fill_int2c2e_ip1 failed')
        # anti-symmetric
        hermi_triu(out, hermi=2, inplace=True)
        out = cell.apply_CT_mat_C(out)
        nao = out.shape[-1]
        out = out.reshape(bvk_ncells, 3, nao, nao)
        if kpts is None:
            out = out[0]
        return out

    def energy_ip1_per_atom(self, dm, kpts=None):
        '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
        if self.bas_ij_cache is None:
            self.build()
        cell = self.cell
        li = np.arange(L_AUX_MAX+1)[:,None]
        lj = np.arange(L_AUX_MAX+1)
        order = li + lj + 1
        nroots = order//2 + 1
        if cell.omega < 0:
            nroots *= 2 # for short-range
        g_size = (li+2)*(lj+2)
        unit = g_size*3 + nroots*2 + 4
        nsp_max = _nearest_power2(SHM_SIZE // (unit*8))
        nsp_per_block = np.where(nsp_max < THREADS, nsp_max, THREADS)
        gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
        shm_size = nsp_per_block * (unit*8)
        lmax = cell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:lmax+1,:lmax+1].max()

        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block)

        nbatches_shl_pair = len(shl_pair_offsets) - 1
        rys_envs = self._rys_envs
        out = cp.zeros((cell.natm, 3))
        libpbc.e_int2c2e_ip1.restype = ctypes.c_int
        err = libpbc.e_int2c2e_ip1(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('e_int2c2e_ip1 failed')
        return out
