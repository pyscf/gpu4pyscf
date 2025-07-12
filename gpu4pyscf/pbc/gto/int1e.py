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
import numpy as np
import cupy as cp
from pyscf.gto import ATOM_OF, PTR_COORD
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, sandwich_dot
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.pbc.df.ft_ao import libpbc, PBCIntEnvVars
from gpu4pyscf.pbc.df.int3c2e import (
    _estimate_shl_pairs_per_block, fill_triu_bvk_conj
)

__all__ = [
    'int1e_ovlp',
    'int1e_ipovlp',
    'int1e_kin',
    'int1e_ipkin',
]

libpbc.PBCint1e_ovlp.restype = ctypes.c_int

LMAX = 4
THREADS = 256
GOUT_WIDTH = 43

def int1e_ovlp(cell, kpts=None, bvk_kmesh=None, opt=None):
    '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
    if opt is None:
        opt = _Int1eOpt(cell, kpts, bvk_kmesh, hermi=1)
    else:
        assert kpts is opt.kpts

    sorted_cell = opt.sorted_cell
    int1e_envs = opt.int1e_envs
    shl_pair_offsets = opt.shl_pair_offsets
    bas_ij_idx = opt.bas_ij_idx
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    nao_cart, nao = opt.coeff.shape
    gout_stride_lookup, shm_size = opt.create_gout_stride_lookup_table()
    bvk_kmesh = opt.bvk_kmesh
    bvk_ncells = np.prod(bvk_kmesh)
    out = cp.empty((bvk_ncells, nao_cart, nao_cart))
    err = libpbc.PBCint1e_ovlp(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(int1e_envs), ctypes.c_int(shm_size),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p),
        sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
        sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
        sorted_cell._env.ctypes)
    if err != 0:
        raise RuntimeError('fill_int2c2e failed')

    out = fill_triu_bvk_conj(out, nao_cart, bvk_kmesh)
    out = sandwich_dot(out, asarray(opt.coeff))

    if kpts is not None:
        bvkmesh_Ls = translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        expLk = cp.exp(1j*asarray(bvkmesh_Ls.dot(kpts.T)))
        out = contract('lk,lpq->kpq', expLk, out)
    return out

def int1e_ipovlp(cell, kpts=None, bvk_kmesh=None):
    pass

def int1e_kin(cell, kpts=None, bvk_kmesh=None):
    pass

def int1e_ipkin(cell, kpts=None, bvk_kmesh=None):
    pass

class _Int1eOpt:
    def __init__(self, cell, kpts=None, bvk_kmesh=None, hermi=1):
        sorted_cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        uniq_l = uniq_l_ctr[:,0]
        lmax = uniq_l.max()
        assert lmax <= LMAX
        self.sorted_cell = sorted_cell
        self.coeff = coeff
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts

        if bvk_kmesh is None:
            if kpts is None:
                bvk_kmesh = np.ones(3, dtype=np.int32)
            else:
                bvk_kmesh = kpts_to_kmesh(cell, kpts)
        self.bvk_kmesh = bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        if bvk_ncells == 1:
            bvkcell = sorted_cell
        else:
            bvkcell = pbctools.super_cell(sorted_cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        Ls = asarray(bvkcell.get_lattice_Ls(rcut=cell.rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)

        _atm = cp.array(bvkcell._atm, dtype=np.int32)
        _bas = cp.array(bvkcell._bas, dtype=np.int32)
        _env = cp.array(_scale_sp_ctr_coeff(bvkcell), dtype=np.float64)
        ao_loc = bvkcell.ao_loc_nr(cart=True)
        ao_loc_gpu = cp.array(ao_loc, dtype=np.int32)
        int1e_envs = PBCIntEnvVars(
            sorted_cell.natm, sorted_cell.nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc_gpu.data.ptr, Ls.data.ptr,
        )
        int1e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)
        self.int1e_envs = int1e_envs

        bas_ij_idx = [] # The effective shell pair = ish*nbas+jsh
        shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
        sp0 = sp1 = 0
        nbas = sorted_cell.nbas
        groups = len(uniq_l)
        if hermi == 1:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.int32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
            img = cp.arange(bvk_ncells, dtype=np.int32)
            ijsh = ish[:,None] * (nbas*bvk_ncells) + jsh
            if hermi and i == j:
                ijsh = ijsh[cp.tril_indices(ish1-ish0)]
            else:
                ijsh = ijsh.ravel()
            idx = (img[:,None] * nbas + ijsh).ravel()
            nshl_pair = len(idx)
            bas_ij_idx.append(idx)
            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = _estimate_shl_pairs_per_block(li, lj, nshl_pair)
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))

        shl_pair_offsets.append(np.array([sp1], dtype=np.int32))
        shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
        bas_ij_idx = cp.array(cp.hstack(bas_ij_idx), dtype=np.int32)
        self.shl_pair_offsets = shl_pair_offsets
        self.bas_ij_idx = bas_ij_idx

    def create_gout_stride_lookup_table(self, lmax=None, deriv=None):
        # based on the shm_size, find optimal gout_stride for each (li,lj)
        # pattern, store them in the gout_stride_lookup
        if lmax is None:
            lmax = self.uniq_l_ctr[:,0].max()
        if deriv is None:
            deriv = (0, 0)
        i_inc, j_inc = deriv
        gout_stride_lookup = np.empty([LMAX+1,LMAX+1], dtype=np.int32)
        gout_width = GOUT_WIDTH # should be identical to the setting fill_int2c2e.cu
        shm_size = SHM_SIZE
        ls = np.arange(lmax+1)
        nf = (ls+1) * (ls+2) // 2
        max_shm_size = 0
        for li in range(lmax+1):
            for lj in range(lmax+1):
                unit = (li+1+i_inc)*(lj+1+i_inc)*3
                nsp_max = _nearest_power2(shm_size // (unit*8))
                gout_size = nf[li] * nf[lj]
                gout_stride = (gout_size+gout_width-1) / gout_width
                # Round up to the next 2^n
                gout_stride = _nearest_power2(gout_stride, return_leq=False)
                nsp_per_block = min(nsp_max, THREADS // gout_stride)
                gout_stride_lookup[li, lj] = THREADS // nsp_per_block
                max_shm_size = max(max_shm_size, nsp_per_block*unit*8)
        return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size


if __name__ == '__main__':
    import pyscf
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3,
        #basis='def2-tzvpp',
        basis=[[0, [1,1]]],
    )

    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    pcell = cell.copy()
    pcell.precision = 1e-14
    pcell.rcut = 50
    ref = pcell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)

    dat = int1e_ovlp(cell).get()[0]
    assert abs(dat - ref[0]).max() < 1e-10

    dat = int1e_ovlp(cell, kpts=kpts).get()
    assert abs(dat - ref).max() < 1e-10
