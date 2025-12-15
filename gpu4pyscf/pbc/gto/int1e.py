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

import math
import ctypes
import numpy as np
import cupy as cp
from pyscf.gto import ATOM_OF, PTR_COORD, Mole
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib.cupy_helper import contract, asarray, sandwich_dot
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import (
    _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE, apply_coeff_C_mat_CT)
from gpu4pyscf.pbc.df.ft_ao import libpbc, PBCIntEnvVars
from gpu4pyscf.pbc.df.int3c2e import (
    _estimate_shl_pairs_per_block, fill_triu_bvk_conj, LMAX, L_AUX_MAX, THREADS
)

__all__ = [
    'int1e_ovlp',
    'int1e_kin',
    'int1e_ipovlp',
    'int1e_ipkin',
]

libpbc.PBCint1e_ovlp.restype = ctypes.c_int
libpbc.PBCint1e_kin.restype = ctypes.c_int
libpbc.PBCint1e_ipovlp.restype = ctypes.c_int
libpbc.PBCint1e_ipkin.restype = ctypes.c_int

def int1e_ovlp(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_ovlp', 1, 1, (0, 0))
    return out

def int1e_kin(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_kin', 1, 1, (2, 0))
    return out

def int1e_ipovlp(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_ipovlp', 0, 3, (1, 0))
    return out

def int1e_ipkin(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_ipkin', 0, 3, (3, 0))
    return out

def _check_opt(cell, kpts, bvk_kmesh, opt):
    if opt is None:
        opt = _Int1eOpt(cell, kpts, bvk_kmesh)
    else:
        assert kpts is None or kpts is opt.kpts
    return opt

class _Int1eOpt:
    def __init__(self, cell, kpts=None, bvk_kmesh=None):
        self.cell = cell
        sorted_cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
        uniq_l = uniq_l_ctr[:,0]
        lmax = uniq_l.max()
        assert lmax <= LMAX
        self.sorted_cell = sorted_cell
        self.coeff = coeff
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts

        if isinstance(cell, Mole):
            # The CUDA code for PBC integrals can be made to support Mole
            # instances. A Mole system can be mimicked by a Gamma point Cell
            # without lattice sum.
            kpts = np.zeros(3)
            bvk_kmesh = np.ones(3, dtype=np.int32)
            bvk_ncells = 1
            bvkcell = sorted_cell
            Ls = cp.zeros((1, 3))
        else:
            if bvk_kmesh is None:
                if kpts is None:
                    bvk_kmesh = np.ones(3, dtype=np.int32)
                else:
                    bvk_kmesh = kpts_to_kmesh(cell, kpts.reshape(-1, 3))
            bvk_ncells = np.prod(bvk_kmesh)
            if bvk_ncells == 1:
                bvkcell = sorted_cell
            else:
                bvkcell = pbctools.super_cell(sorted_cell, bvk_kmesh, wrap_around=True)
                # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
                bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            Ls = asarray(bvkcell.get_lattice_Ls(rcut=cell.rcut))
            Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]

        self.kpts = kpts
        self.bvk_kmesh = bvk_kmesh
        self.bvkcell = bvkcell
        nimgs = len(Ls)

        _atm = cp.array(bvkcell._atm, dtype=np.int32)
        _bas = cp.array(bvkcell._bas, dtype=np.int32)
        _env = cp.array(_scale_sp_ctr_coeff(bvkcell), dtype=np.float64)
        ao_loc = bvkcell.ao_loc_nr(cart=True)
        ao_loc_gpu = cp.array(ao_loc, dtype=np.int32)
        self.int1e_envs = PBCIntEnvVars.new(
            sorted_cell.natm, sorted_cell.nbas, bvk_ncells, nimgs,
            _atm, _bas, _env, ao_loc_gpu, Ls)

    def generate_shl_pairs(self, hermi, gout_stride_lookup):
        sorted_cell = self.sorted_cell
        l_ctr_offsets = np.append(0, np.cumsum(self.l_ctr_counts))
        uniq_l = self.uniq_l_ctr[:,0]
        bas_ij_idx = [] # The effective shell pair = ish*nbas+jsh
        shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
        sp0 = sp1 = 0
        nbas = sorted_cell.nbas
        groups = len(uniq_l)
        if hermi == 1:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
        bvk_ncells = np.prod(self.bvk_kmesh)
        img = cp.arange(bvk_ncells, dtype=np.int32)
        img_offsets = img * nbas
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.int32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
            ijsh = ish[:,None] * (nbas*bvk_ncells) + jsh
            if hermi and i == j:
                ijsh = ijsh[cp.tril_indices(ish1-ish0)]
            else:
                ijsh = ijsh.ravel()
            idx = (img_offsets[:,None] + ijsh).ravel()
            nshl_pair = len(idx)
            bas_ij_idx.append(idx)
            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = THREADS // gout_stride_lookup[li, lj] * 8
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))

        shl_pair_offsets.append(np.int32(sp1))
        shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
        bas_ij_idx = cp.array(cp.hstack(bas_ij_idx), dtype=np.int32)
        return bas_ij_idx, shl_pair_offsets

    def create_gout_stride_lookup_table(self, deriv=None, gout_width=36):
        # gout_width should be identical to the setting in cuda kernel
        # based on the shm_size, find optimal gout_stride for each (li,lj)
        # pattern, store them in the gout_stride_lookup
        if deriv is None:
            deriv = (0, 0)
        i_inc, j_inc = deriv
        lmax = self.uniq_l_ctr[:,0].max()
        gout_stride_lookup = np.empty([L_AUX_MAX+1,L_AUX_MAX+1], dtype=np.int32)
        shm_size = SHM_SIZE
        ls = np.arange(lmax+1)
        nf = (ls+1) * (ls+2) // 2
        max_shm_size = 0
        for li in range(lmax+1):
            for lj in range(lmax+1):
                unit = (li+1+i_inc)*(lj+1+j_inc)*3 + 4
                nsp_max = _nearest_power2(shm_size // (unit*8))
                gout_size = nf[li] * nf[lj]
                gout_stride = (gout_size+gout_width-1) / gout_width
                # Round up to the next 2^n
                gout_stride = _nearest_power2(gout_stride, return_leq=False)
                nsp_per_block = min(nsp_max, THREADS // gout_stride)
                gout_stride_lookup[li, lj] = THREADS // nsp_per_block
                max_shm_size = max(max_shm_size, nsp_per_block*unit*8)
        return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size

    def intor(self, kern, hermi, comp, deriv_ij):
        if comp == 1:
            gout_width = 36
        else:
            gout_width = 18

        sorted_cell = self.sorted_cell
        gout_stride_lookup, shm_size = self.create_gout_stride_lookup_table(
            deriv_ij, gout_width)
        bas_ij_idx, shl_pair_offsets = self.generate_shl_pairs(hermi, gout_stride_lookup)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        nao_cart, nao = self.coeff.shape
        out = cp.empty((bvk_ncells, comp, nao_cart, nao_cart))
        int1e_envs = self.int1e_envs
        drv = getattr(libpbc, kern)
        err = drv(
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
            raise RuntimeError(f'{kern} failed')

        if hermi == 1:
            assert comp == 1
            out = fill_triu_bvk_conj(out, nao_cart, bvk_kmesh)
        out = sandwich_dot(out.reshape(-1,nao_cart,nao_cart), self.coeff)
        out = out.reshape(bvk_ncells, comp, nao, nao)

        if self.kpts is not None and not is_zero(self.kpts):
            bvkmesh_Ls = translation_vectors_for_kmesh(self.cell, bvk_kmesh, True)
            kpts = self.kpts.reshape(-1, 3)
            expLk = cp.exp(1j*asarray(bvkmesh_Ls.dot(kpts.T)))
            out = contract('lk,lxpq->kxpq', expLk, out)

        if comp == 1:
            out = out[:,0]
        if self.kpts is not None and self.kpts.ndim == 1:
            out = out[0]
        return out

class _Int1eOptV2:
    def __init__(self, cell):
        self.cell = cell
        cell, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = group_basis(
            cell, 1, sparse_coeff=True)
        lmax = uniq_l_ctr[:,0].max()
        assert lmax <= L_AUX_MAX
        self.sorted_cell = cell
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts
        self.l_ctr_pad_counts = l_ctr_pad_counts
        self.ao_idx = ao_idx

        if isinstance(cell, Mole):
            # The CUDA code for PBC integrals can be made to support Mole instances.
            Ls = np.zeros((1, 3))
            self.precision = 1e-16
        else:
            Ls = cell.get_lattice_Ls()
            Ls = Ls[np.linalg.norm(Ls-.1, axis=1).argsort()]
            self.precision = cell.precision * 1e-4
        self.Ls = Ls
        self._int1e_envs = {}

    @multi_gpu.property(cache='_int1e_envs')
    def int1e_envs(self):
        cell = self.sorted_cell
        atm = asarray(cell._atm)
        bas = asarray(cell._bas)
        env = asarray(_scale_sp_ctr_coeff(cell))
        ao_loc = asarray(cell.ao_loc)
        Ls = asarray(self.Ls)
        nimgs = len(Ls)
        return PBCIntEnvVars.new(cell.natm, cell.nbas, nimgs, nimgs, atm, bas, env, ao_loc, Ls)

    def generate_shl_pairs(self, hermi=1, gout_stride_lookup=None):
        sorted_cell = self.sorted_cell
        ovlp_mask = _shell_overlap_mask(sorted_cell, hermi, self.precision, self.Ls)

        pairs_idx = cp.arange(ovlp_mask.size, dtype=np.int32)
        pairs_idx = pairs_idx.reshape(ovlp_mask.shape)
        l_ctr_offsets = np.append(0, np.cumsum(self.l_ctr_counts))
        uniq_l = self.uniq_l_ctr[:,0]
        bas_ij_idx = [] # The effective shell pair = ish*nbas+jsh
        shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
        sp0 = sp1 = 0
        groups = len(uniq_l)
        if hermi == 1:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
            i, j = cp.triu_indices(sorted_cell.nbas, 1)
            ovlp_mask[i,:,j] = False
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            mask = ovlp_mask[ish0:ish1,:,jsh0:jsh1]
            idx = pairs_idx[ish0:ish1,:,jsh0:jsh1][mask]
            nshl_pair = len(idx)
            bas_ij_idx.append(idx)
            sp0, sp1 = sp1, sp1 + nshl_pair
            if gout_stride_lookup is None:
                nsp_per_block = 512
            else:
                nsp_per_block = THREADS // gout_stride_lookup[li, lj] * 8
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))

        shl_pair_offsets.append(np.int32(sp1))
        shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
        bas_ij_idx = cp.array(cp.hstack(bas_ij_idx), dtype=np.int32)
        return bas_ij_idx, shl_pair_offsets

    def create_gout_stride_lookup_table(self, deriv=None, gout_width=36):
        # gout_width should be identical to the setting in cuda kernel
        # based on the shm_size, find optimal gout_stride for each (li,lj)
        # pattern, store them in the gout_stride_lookup
        if deriv is None:
            deriv = (0, 0)
        i_inc, j_inc = deriv
        lmax = self.uniq_l_ctr[:,0].max()
        gout_stride_lookup = np.empty([L_AUX_MAX+1,L_AUX_MAX+1], dtype=np.int32)
        shm_size = SHM_SIZE
        ls = np.arange(lmax+1)
        nf = (ls+1) * (ls+2) // 2
        max_shm_size = 0
        for li in range(lmax+1):
            for lj in range(lmax+1):
                unit = (li+1+i_inc)*(lj+1+j_inc)*3 + 4
                nsp_max = _nearest_power2(shm_size // (unit*8))
                gout_size = nf[li] * nf[lj]
                gout_stride = (gout_size+gout_width-1) / gout_width
                # Round up to the next 2^n
                gout_stride = _nearest_power2(gout_stride, return_leq=False)
                nsp_per_block = min(nsp_max, THREADS // gout_stride)
                gout_stride_lookup[li, lj] = THREADS // nsp_per_block
                max_shm_size = max(max_shm_size, nsp_per_block*unit*8)
        return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size

    def get_ovlp_strain_deriv(self, dm, kpts=None):
        '''Computes the strain derivatives for the product of density matrix and
        overlap matrix. In the case of k-points calculations, the derivatives
        are averaged over k-mesh.
        '''
        cell = self.cell
        sorted_cell = self.sorted_cell
        nao_orig = cell.nao
        dm = asarray(dm, order='C')
        dm = dm.reshape(-1,nao_orig,nao_orig)
        l_ctr_offsets = np.append(0, np.cumsum(self.l_ctr_counts))
        dm = apply_coeff_C_mat_CT(dm, cell, sorted_cell, self.uniq_l_ctr,
                                  l_ctr_offsets, self.ao_idx)
        if kpts is None:
            kpts = np.zeros((1, 3))
        else:
            kpts = kpts.reshape(-1, 3)
        nkpts = len(kpts)
        nao = dm.shape[-1]
        dm = dm.reshape(-1, nkpts, nao, nao)
        if len(dm) == 1:
            dm = dm[0]
        else:
            dm = dm.sum(axis=0)

        is_gamma_point = is_zero(kpts)
        if is_gamma_point:
            assert dm.dtype == np.float64
        else:
            expLk = cp.exp(1j * asarray(self.Ls).dot(asarray(kpts).T))
            dm = contract('Lk,kpq->Lpq', expLk, dm)
            expLk = None
            dm = dm.real
        dm = cp.asarray(dm, order='C')

        hermi = 1
        deriv = (1, 0)
        gout_stride_lookup, shm_size = self.create_gout_stride_lookup_table(deriv)
        bas_ij_idx, shl_pair_offsets = self.generate_shl_pairs(hermi, gout_stride_lookup)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        int1e_envs = self.int1e_envs
        sigma = cp.zeros((3, 3))
        libpbc.PBCovlp_strain_deriv(
            ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(int1e_envs),
            ctypes.c_int(shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p),
            ctypes.c_int(int(is_gamma_point)))
        sigma = sigma.get()
        sigma *= 2 / nkpts
        return sigma

def _shell_overlap_mask(mol, hermi=1, precision=1e-14, Ls=None):
    '''absmax(<i|j>) > precision for each shell pair'''
    nbas = mol.nbas
    exps, cs = extract_pgto_params(mol, 'diffuse')
    exps = cp.asarray(exps, dtype=np.float32)
    log_coeff = cp.log(abs(asarray(cs, dtype=np.float32)))
    ao_loc = cp.arange(0)
    with_images = Ls is not None
    if Ls is None:
        Ls = cp.zeros((1, 3))
    else:
        Ls = asarray(Ls)
    nimgs = len(Ls)
    ovlp_mask = cp.zeros((nbas,nimgs,nbas), dtype=bool)
    envs = PBCIntEnvVars.new(
        mol.natm, mol.nbas, nimgs, nimgs, asarray(mol._atm),
        asarray(mol._bas), asarray(_scale_sp_ctr_coeff(mol)), ao_loc, Ls)
    libpbc.PBCovlp_mask_estimation(
        ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
        ctypes.cast(exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
        ctypes.byref(envs), ctypes.c_int(hermi),
        ctypes.c_float(math.log(precision)))
    if not with_images:
        ovlp_mask = ovlp_mask[:,0]
    return ovlp_mask
