# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
Perodic 3-center 2-electron short-range Coulomb integral helper functions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf.gto import (ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
                       PTR_COORD, BAS_SLOTS, conc_env)
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, sandwich_dot
from gpu4pyscf.gto.mole import (cart2sph_by_l, group_basis, PTR_BAS_COORD,
                                extract_pgto_params)
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.pbc.df.ft_ao import (
    libpbc, init_constant, most_diffused_pgto, PBCIntEnvVars)
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell

__all__ = [
    'sr_aux_e2',
]

libpbc.fill_int3c2e.restype = ctypes.c_int
libpbc.fill_int2c2e.restype = ctypes.c_int
libpbc.bvk_overlap_img_idx.restype = ctypes.c_int
libpbc.sr_int3c2e_img_idx.restype = ctypes.c_int
libpbc.conc_img_idx.restype = ctypes.c_int
libpbc.aopair_fill_triu.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 45
THREADS = 256

def sr_aux_e2(cell, auxcell, omega, kpts=None, bvk_kmesh=None, j_only=False):
    r'''
    Short-range 3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    if bvk_kmesh is None and kpts is not None:
        if j_only:
            # Coulomb integrals requires smaller kmesh to converge finite-size effects
            bvk_kmesh = kpts_to_kmesh(cell, kpts)
        else:
            # The remote images may contribute to certain k-point mesh,
            # contributing to the finite-size effects in exchange matrix.
            rcut = estimate_rcut(cell, auxcell, omega).max()
            bvk_kmesh = kpts_to_kmesh(cell, kpts, rcut=rcut)

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh).build()
    nao = cell.nao
    naux = int3c2e_opt.aux_coeff.shape[1]

    gamma_point = kpts is None or (kpts.ndim == 1 and is_zero(kpts))
    if gamma_point:
        out = cp.zeros((nao, nao, naux))
        nL = nkpts = 1
    else:
        kpts = np.asarray(kpts).reshape(-1, 3)
        expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
        nL, nkpts = expLk.shape
        if j_only:
            expLLk = contract('Mk,Lk->MLk', expLk.conj(), expLk)
            expLLk = expLLk.view(np.float64).reshape(nL,nL,nkpts,2)
            out = cp.zeros((nkpts, nao, nao, naux), dtype=np.complex128)
        else:
            out = cp.zeros((nkpts, nkpts, nao, nao, naux), dtype=np.complex128)

    c_shell_counts = np.asarray(int3c2e_opt.cell0_ctr_l_counts)
    lmax = len(c_shell_counts) - 1
    uniq_l = np.arange(lmax+1)
    if cell.cart:
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
    else:
        nf = uniq_l * 2 + 1
    c_l_offsets = np.append(0, np.cumsum(c_shell_counts*nf))
    c2s = [cart2sph_by_l(l) for l in range(lmax+1)]

    aux_coeff = asarray(int3c2e_opt.aux_coeff)
    for li, lj, c_pair_idx, compressed_eri3c in int3c2e_opt.int3c2e_generator():
        i0, i1 = c_l_offsets[li:li+2]
        j0, j1 = c_l_offsets[lj:lj+2]
        nctri = c_shell_counts[li]
        nctrj = c_shell_counts[lj]
        nfi = (li+1)*(li+2)//2
        nfj = (lj+1)*(lj+2)//2
        nfij = nfi * nfj
        n_pairs = len(c_pair_idx)
        compressed_eri3c = compressed_eri3c.reshape(-1,nfij*n_pairs)
        compressed_eri3c = compressed_eri3c.T.dot(aux_coeff)
        if not cell.cart:
            compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs,naux)
            compressed_eri3c = contract('qj,qpmk->jpmk', c2s[lj], compressed_eri3c)
            compressed_eri3c = contract('pi,jpmk->jimk', c2s[li], compressed_eri3c)
            nfi = li * 2 + 1
            nfj = lj * 2 + 1

        ni = i1 - i0
        nj = j1 - j0
        ish, jsh = divmod(c_pair_idx, nL*nctrj)
        eri3c = cp.zeros((nL*nctri,nfi, nL*nctrj,nfj, naux))
        compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs,naux)
        eri3c[ish,:,jsh] = compressed_eri3c.transpose(2,1,0,3)
        if i0 == j0:
            eri3c[jsh,:,ish] = compressed_eri3c.transpose(2,0,1,3)
        eri3c = eri3c.reshape(nL,ni,nL,nj,naux)
        compressed_eri3c = None

        i = int3c2e_opt.ao_idx[i0:i1]
        j = int3c2e_opt.ao_idx[j0:j1]
        if gamma_point:
            eri3c = eri3c.reshape(ni,nj,naux)
            out[i[:,None],j] = eri3c
            if i0 != j0:
                out[j[:,None],i] = eri3c.transpose(1,0,2)
        elif j_only:
            eri3c = contract('MLkz,MpLqr->kpqrz', expLLk, eri3c)
            eri3c = eri3c.view(np.complex128)[...,0]
            out[:,i[:,None],j] = eri3c
            if i0 != j0:
                out[:,j[:,None],i] = eri3c.transpose(0,2,1,3).conj()
        else:
            expLkz = expLk.view(np.float64).reshape(nL,nkpts,2)
            eri3c = contract('Lkz,MpLqr->Mkpqrz', expLkz, eri3c)
            eri3c = eri3c.view(np.complex128)[...,0]
            eri3c = contract('Mk,Mlpqr->klpqr', expLk.conj(), eri3c)
            out[:,:,i[:,None],j] = eri3c
            if i0 != j0:
                out[:,:,j[:,None],i] = eri3c.transpose(1,0,3,2,4).conj()
        eri3c = None
    return out

def sr_int2c2e(cell, omega, kpts=None, bvk_kmesh=None):
    '''SR 2c2e Coulomb integrals for the auxiliary basis set'''
    assert omega < 0
    assert cell._bas[:,ANG_OF].max() <= L_AUX_MAX

    sorted_cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    sorted_cell.omega = omega
    uniq_l = uniq_l_ctr[:,0]
    lmax = uniq_l.max()

    if bvk_kmesh is None:
        if kpts is None:
            bvk_kmesh = np.ones(3, dtype=np.int32)
        else:
            bvk_kmesh = kpts_to_kmesh(cell, kpts)
    bvk_ncells = np.prod(bvk_kmesh)
    if bvk_ncells == 1:
        bvkcell = sorted_cell
    else:
        bvkcell = pbctools.super_cell(sorted_cell, bvk_kmesh, wrap_around=True)
        # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
        bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]

    precision = cell.precision * 1e-3
    ak, ck, lk = most_diffused_pgto(sorted_cell)
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

    Ls = asarray(bvkcell.get_lattice_Ls(rcut=rcut))
    Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
    nimgs = len(Ls)

    _atm = cp.array(bvkcell._atm, dtype=np.int32)
    _bas = cp.array(bvkcell._bas, dtype=np.int32)
    _env = cp.array(_scale_sp_ctr_coeff(bvkcell), dtype=np.float64)
    ao_loc = bvkcell.ao_loc_nr(cart=True)
    ao_loc_gpu = cp.array(ao_loc, dtype=np.int32)
    int3c2e_envs = PBCIntEnvVars(
        sorted_cell.natm, sorted_cell.nbas, bvk_ncells, nimgs,
        _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
        ao_loc_gpu.data.ptr, Ls.data.ptr,
    )

    bas_ij_idx = [] # The effective shell pair = ish*nbas+jsh
    shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
    sp0 = sp1 = 0
    nbas = sorted_cell.nbas
    ij_tasks = [(i, j) for i in range(len(uniq_l)) for j in range(i+1)]
    for i, j in ij_tasks:
        li = uniq_l[i]
        lj = uniq_l[j]
        ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
        jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
        ish = cp.arange(ish0, ish1, dtype=np.int32)
        jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
        img = cp.arange(bvk_ncells, dtype=np.int32)
        ijsh = ish[:,None] * (nbas*bvk_ncells) + jsh
        if i == j:
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

    def _create_gout_stride_lookup_table(lmax):
        # based on the shm_size, find optimal gout_stride for each (li,lj)
        # pattern, store them in the gout_stride_lookup
        gout_stride_lookup = np.empty([L_AUX_MAX+1,L_AUX_MAX+1], dtype=np.int32)
        gout_width = 43 # should be identical to the setting fill_int2c2e.cu
        shm_size = SHM_SIZE
        ls = np.arange(lmax+1)
        nf = (ls+1) * (ls+2) // 2
        max_shm_size = 0
        for li in range(lmax+1):
            for lj in range(lmax+1):
                nroots = ((li + lj) // 2 + 1) * 2
                g_size = (li+1)*(lj+1)
                unit = g_size*3 + nroots*2 + 4
                nsp_max = _nearest_power2(shm_size // (unit*8))

                gout_size = nf[li] * nf[lj]
                gout_stride = (gout_size+gout_width-1) / gout_width
                # Round up to the next 2^n
                gout_stride = _nearest_power2(gout_stride, return_leq=False)

                nsp_per_block = min(nsp_max, THREADS // gout_stride)
                gout_stride_lookup[li, lj] = THREADS // nsp_per_block
                max_shm_size = max(max_shm_size, nsp_per_block*unit*8)
        return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size

    gout_stride_lookup, shm_size = _create_gout_stride_lookup_table(lmax)

    nbatches_shl_pair = len(shl_pair_offsets) - 1
    nao_cart, nao = coeff.shape
    out = cp.empty((bvk_ncells, nao_cart, nao_cart))
    init_constant(cell)
    err = libpbc.fill_int2c2e(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size),
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
    out = sandwich_dot(out, asarray(coeff))

    if kpts is not None:
        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        expLk = cp.exp(1j*asarray(bvkmesh_Ls.dot(kpts.T)))
        out = contract('lk,lpq->kpq', expLk, out)
    return out

def fill_triu_bvk_conj(a, nao, bvk_kmesh):
    # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
    assert a.flags.c_contiguous
    conj_mapping = conj_images_in_bvk_cell(bvk_kmesh)
    conj_mapping = cp.asarray(conj_mapping, dtype=np.int32)
    bvk_ncells = np.prod(bvk_kmesh)
    err = libpbc.aopair_fill_triu(
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(conj_mapping.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(bvk_ncells))
    if err != 0:
        raise RuntimeError('aopair_fill_triu failed')
    return a

def to_primitive_bas(cell):
    '''Decontract the cell basis sets into primitive bases'''
    bas_templates = {}
    prim_bas = []
    prim_env = cell._env.copy()
    shell_offset = 0
    # Mapping from the primitive shell to the shell in the original cell.
    prim_to_ctr_mapping = []
    aoslices = cell.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        ptr_coord = cell._atm[ia,PTR_COORD]
        key = tuple(cell._bas[ib0:ib1,PTR_COEFF])
        if key in bas_templates:
            bas_of_ia, local_shell_mapping = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,ATOM_OF] = ia
            bas_of_ia[:,PTR_BAS_COORD] = ptr_coord
        else:
            # Generate the template for decontracted basis
            local_shell_mapping = []
            off = 0
            bas_of_ia = []
            for shell in cell._bas[ib0:ib1]:
                l = shell[ANG_OF]
                nctr = shell[NCTR_OF]
                nprim = shell[NPRIM_OF]
                pexp = shell[PTR_EXP]
                pcoeff = shell[PTR_COEFF]
                bs = np.empty((nprim*nctr, BAS_SLOTS), dtype=np.int32)
                bs[:,ATOM_OF] = ia
                bs[:,ANG_OF] = l
                bs[:,NPRIM_OF] = 1
                bs[:,NCTR_OF] = 1
                bs[:,PTR_EXP] = np.hstack([np.arange(pexp, pexp+nprim)] * nctr)
                bs[:,PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr)
                bs[:,PTR_BAS_COORD] = ptr_coord
                bas_of_ia.append(bs)
                idx = np.repeat(np.arange(off, off+nctr), nprim)
                local_shell_mapping.append(idx)
                off += nctr

            '''TODO
            # partition the contracted GTO into a compact subset and
            # multiple primitive shells
            for shell in cell._bas[ib0:ib1]:
                l = shell[ANG_OF]
                nprim = shell[NPRIM_OF]
                nctr = shell[NCTR_OF]
                pexp = shell[PTR_EXP]
                es = prim_env[pexp:pexp+nprim]
                diffused_idx = np.where(es < 2.)[0]
                n_diffused = len(diffuse_idx)
                for ic in range(nctr):
                    pcoeff = shell[PTR_COEFF] + ic * nprim
                    bs = shell.copy()
                    bs[NCTR_OF] = 1
                    bs[PTR_COEFF] = pcoeff
                    bs[PTR_BAS_COORD] = ptr_coord
                    if nprim == 1 or n_diffused == 0:
                        bas_of_ia.append(bs)
                        local_shell_mapping.append(off+ic)
                        continue

                    cs = prim_env[pcoeff:pcoeff+nprim]
                    compact_idx = np.where(es >= 2)[0]
                    n_compact = len(compact_idx)
                    idx = np.hstack(compact_idx, diffuse_idx)
                    prim_env[pexp:pexp+nprim] = es[idx]
                    prim_env[pcoeff:pcoeff+n_compact] = cs[compact_idx]
                    prim_env[pcoeff+n_compact:pcoeff+nprim] = cs[diffused_idx]
                    if n_compact > 0:
                        # put compact pGTOs in one shell
                        bs[NPRIM_OF] = n_compact
                        bas_of_ia.append(bs.copy())
                        local_shell_mapping.append(off+ic)
                        pexp += n_compact
                        pcoeff += n_compact
                    # each diffused pGTO as one shell
                    bs[NPRIM_OF] = 1
                    for m in range(n_diffused):
                        bs[PTR_EXP] = pexp + m
                        bs[PTR_COEFF] = pexp + m
                        bas_of_ia.append(bs.copy())
                        local_shell_mapping.append(off+ic)
                off += nctr
                '''

            if bas_of_ia:
                bas_of_ia = np.vstack(bas_of_ia)
                local_shell_mapping = np.hstack(local_shell_mapping)
                bas_templates[key] = (bas_of_ia, local_shell_mapping)

        if len(bas_of_ia) > 0:
            prim_bas.append(bas_of_ia)
            prim_to_ctr_mapping.append(shell_offset + local_shell_mapping)
            shells_in_atm = cell._bas[ib0:ib1,NCTR_OF].sum()
            shell_offset += shells_in_atm

    pcell = cell.copy()
    pcell._bas = np.asarray(np.vstack(prim_bas), dtype=np.int32)
    pcell._env = prim_env
    prim_to_ctr_mapping = np.asarray(np.hstack(prim_to_ctr_mapping), dtype=np.int32)

    p_ls = pcell._bas[:,ANG_OF]
    lmax = p_ls.max()
    sorted_idx = np.hstack([np.where(p_ls==l)[0] for l in range(lmax+1)])
    pcell._bas = pcell._bas[sorted_idx]

    # This sorted_cell is a fictitious cell object, to define the
    # p2c_mapping for prim_cell. PTRs in sorted_cell are not initialized.
    # This object should not be used for any integral kernel.
    sorted_cell = cell.copy()
    c_ls = np.repeat(cell._bas[:,ANG_OF], cell._bas[:,NCTR_OF])
    sorted_idx = np.repeat(np.arange(cell.nbas), cell._bas[:,NCTR_OF])
    sorted_idx = [sorted_idx[c_ls==l] for l in range(lmax+1)]
    counts = [len(i) for i in sorted_idx]
    sorted_idx = np.hstack(sorted_idx)
    sorted_cell._bas = cell._bas[sorted_idx]
    sorted_cell._bas[:,NCTR_OF] = 1

    # prim shells are sorted in pcell. The mapping needs to be sorted accordingly.
    # The lookup stores the mapping for each angular momentum
    c_shell_offsets = np.append(0, np.cumsum(counts))
    p2c_mapping = []
    for l, offset in enumerate(c_shell_offsets[:-1]):
        i, idx = np.unique(prim_to_ctr_mapping[p_ls==l], return_inverse=True)
        assert all(i[:-1] < i[1:])
        p2c_mapping.append(idx + offset)
    p2c_mapping = np.asarray(np.hstack(p2c_mapping), dtype=np.int32)

    # ao_idx transforms the AOs in sorted_cell into AOs in the original cell
    if cell.cart:
        dims = (c_ls + 1) * (c_ls + 2) // 2
    else:
        dims = c_ls * 2+ 1
    ao_loc = np.append(np.int32(0), dims.cumsum(dtype=np.int32))
    idx = np.hstack([np.where(c_ls==l)[0] for l in range(lmax+1)])
    ao_idx = np.array_split(np.arange(cell.nao), ao_loc[1:-1])
    ao_idx = np.hstack([ao_idx[i] for i in idx])
    return pcell, sorted_cell, p2c_mapping, ao_idx

class SRInt3c2eOpt:
    def __init__(self, cell, auxcell, omega, bvk_kmesh=None):
        assert omega < 0
        self.omega = -omega
        assert cell._bas[:,ANG_OF].max() <= LMAX

        self.cell = cell
        prim_cell, sorted_cell, self.prim_to_ctr_mapping, self.ao_idx = \
                to_primitive_bas(cell)
        self.prim_cell = prim_cell
        self.prim_cell.omega = omega
        # This sorted_cell is a fictitious cell object, to define the
        # p2c_mapping for prim_cell. PTRs in sorted_cell are not initialized.
        # This object should not be used for any integral kernel.
        self.sorted_cell = sorted_cell

        self.cell0_prim_l_counts = np.bincount(prim_cell._bas[:,ANG_OF])
        self.cell0_ctr_l_counts = np.bincount(sorted_cell._bas[:,ANG_OF])

        self.auxcell = auxcell
        auxcell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(auxcell, tile=1)
        self.sorted_auxcell = auxcell
        self.uniq_l_ctr_aux = uniq_l_ctr
        self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.aux_coeff = coeff
        self.sorted_auxcell.omega = omega

        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh

        self.rcut = None
        self.int3c2e_envs = None
        self.bvk_cell = None
        self.bvkmesh_Ls = None

    def build(self, verbose=None):
        '''integral screening'''
        log = logger.new_logger(self.cell, verbose)
        pcell = self.prim_cell
        auxcell = self.sorted_auxcell

        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        self.bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(pcell, bvk_kmesh, True)
        if np.prod(bvk_kmesh) == 1:
            bvkcell = pcell
        else:
            bvkcell = pbctools.super_cell(pcell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        self.rcut = rcut = estimate_rcut(pcell, auxcell, self.omega).max()
        Ls = asarray(bvkcell.get_lattice_Ls(rcut=rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug('int3c2e_kernel rcut = %g, nimgs = %d', rcut, nimgs)

        # Note: sort_orbitals and unsort_orbitals do not transform the
        # s and p orbitals. _scale_sp_ctr_coeff apply these special
        # normalization coefficients to the _env.
        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            auxcell._atm, auxcell._bas, _scale_sp_ctr_coeff(auxcell))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[bvkcell.nbas,PTR_EXP] - auxcell._bas[0,PTR_EXP]
        _bas_cpu[bvkcell.nbas:,PTR_BAS_COORD] += off
        self._atm_cpu = _atm_cpu
        self._bas_cpu = _bas_cpu
        self._env_cpu = _env_cpu

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        bvk_ao_loc = bvkcell.ao_loc
        aux_loc = auxcell.ao_loc
        ao_loc = _conc_locs(bvk_ao_loc, aux_loc)
        int3c2e_envs = PBCIntEnvVars(
            pcell.natm, pcell.nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc.data.ptr, Ls.data.ptr,
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)
        self.int3c2e_envs = int3c2e_envs
        init_constant(pcell)

        log.debug1('prim_l_counts %s', self.cell0_prim_l_counts)
        log.debug1('ctr_l_counts %s', self.cell0_ctr_l_counts)
        return self

    def estimate_cutoff_with_penalty(self):
        pcell = self.prim_cell
        auxcell = self.sorted_auxcell
        vol = self.bvkcell.vol
        omega = self.omega
        aux_exp, _, aux_l = most_diffused_pgto(auxcell)
        cell_exp, _, cell_l = most_diffused_pgto(pcell)
        if omega == 0:
            theta = 1./(1./cell_exp*2 + 1./aux_exp)
        else:
            theta = 1./(1./cell_exp*2 + 1./aux_exp + omega**-2)
        lsum = cell_l * 2 + aux_l + 1
        rad = vol**(-1./3) * self.rcut + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = 2*np.pi*self.rcut*lsum/(vol*theta) + surface
        cutoff = pcell.precision / lattice_sum_factor
        logger.debug1(pcell, 'int3c_kernel integral omega=%g theta=%g cutoff=%g',
                      omega, theta, cutoff)
        return cutoff

    def generate_img_idx(self, cutoff=None, verbose=None):
        log = logger.new_logger(self.cell, verbose)
        cput0 = log.init_timer()
        int3c2e_envs = self.int3c2e_envs
        pcell = self.prim_cell
        auxcell = self.sorted_auxcell
        bvk_ncells = np.prod(self.bvk_kmesh)
        p_nbas = pcell.nbas

        exps, cs = extract_pgto_params(pcell, 'diffused')
        exps = asarray(exps, dtype=np.float32)
        log_coeff = cp.log(abs(asarray(cs, dtype=np.float32)))

        # Search the most diffused functions on each atom
        aux_exps, aux_cs = extract_pgto_params(auxcell, 'diffused')
        aux_ls = auxcell._bas[:,ANG_OF]
        r2_aux = np.log(aux_cs**2 / pcell.precision * 10**aux_ls) / aux_exps
        atoms = auxcell._bas[:,ATOM_OF]
        atom_aux_exps = np.full(pcell.natm, 1e8, dtype=np.float32)
        for ia in range(pcell.natm):
            bas_mask = atoms == ia
            es = aux_exps[bas_mask]
            if len(es) > 0:
                atom_aux_exps[ia] = es[r2_aux[bas_mask].argmax()]
        atom_aux_exps = asarray(atom_aux_exps, dtype=np.float32)
        if cutoff is None:
            cutoff = self.estimate_cutoff_with_penalty()
        log_cutoff = math.log(cutoff)

        c_shell_counts = self.cell0_ctr_l_counts
        c_shell_offsets = np.append(0, np.cumsum(c_shell_counts))
        p_shell_l_offsets = np.append(0, np.cumsum(self.cell0_prim_l_counts))
        p2c_mapping = asarray(self.prim_to_ctr_mapping, dtype=np.int32)

        def gen_img_idx(li, lj):
            t0 = log.init_timer()
            ish0, ish1 = p_shell_l_offsets[li:li+2]
            jsh0, jsh1 = p_shell_l_offsets[lj:lj+2]
            nprimi = ish1 - ish0
            nprimj = jsh1 - jsh0
            nctri = c_shell_counts[li]
            nctrj = c_shell_counts[lj]

            # Number of images for each pair of (bas_i_in_bvkcell, bas_j_in_bvkcell)
            ovlp_img_counts = cp.zeros((bvk_ncells*nprimi*bvk_ncells*nprimj), dtype=np.int32)
            err = libpbc.bvk_overlap_img_counts(
                ctypes.cast(ovlp_img_counts.data.ptr, ctypes.c_void_p),
                ctypes.cast(p2c_mapping.data.ptr, ctypes.c_void_p),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('bvk_overlap_img_counts failed')

            bas_ij = asarray(cp.where(ovlp_img_counts > 0)[0], dtype=np.int32)
            ovlp_npairs = len(bas_ij)
            if ovlp_npairs == 0:
                img_idx = offsets = bas_ij = pair_mapping = c_pair_idx = np.zeros(0, dtype=np.int32)
                return img_idx, offsets, bas_ij, pair_mapping, c_pair_idx

            counts_sorting = (-ovlp_img_counts[bas_ij]).argsort()
            bas_ij = bas_ij[counts_sorting]
            ovlp_img_counts = ovlp_img_counts[bas_ij]
            ovlp_img_offsets = cp.empty(ovlp_npairs+1, dtype=np.int32)
            ovlp_img_offsets[0] = 0
            cp.cumsum(ovlp_img_counts, out=ovlp_img_offsets[1:])
            tot_imgs = int(ovlp_img_offsets[ovlp_npairs])
            ovlp_img_idx = cp.empty(tot_imgs, dtype=np.int32)
            err = libpbc.bvk_overlap_img_idx(
                ctypes.cast(ovlp_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ovlp_npairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('bvk_overlap_img_idx failed')
            log.timer_debug1('ovlp_img_idx', *cput0)
            nimgs_J = int(ovlp_img_counts[0])
            ovlp_img_counts = counts_sorting = None

            img_counts = cp.zeros(ovlp_npairs, dtype=np.int32)
            ovlp_pair_sorting = cp.arange(len(bas_ij), dtype=np.int32)
            err = libpbc.sr_int3c2e_img_idx(
                lib.c_null_ptr(),
                ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_pair_sorting.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ovlp_npairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('sr_int3c2e_img_counts failed')

            n_pairs = int(cp.count_nonzero(img_counts))
            if n_pairs == 0:
                img_idx = offsets = bas_ij = pair_mapping = c_pair_idx = np.zeros(0, dtype=np.int32)
                return img_idx, offsets, bas_ij, pair_mapping, c_pair_idx

            # Sorting the bas_ij pairs by image counts. This groups bas_ij into
            # groups with similar workloads in int3c2e kernel.
            counts_sorting = cp.argsort(-img_counts.ravel())[:n_pairs]
            counts_sorting = asarray(counts_sorting, dtype=np.int32)
            bas_ij = bas_ij[counts_sorting]
            ovlp_pair_sorting = counts_sorting
            img_counts = img_counts[counts_sorting]
            offsets = cp.empty(n_pairs+1, dtype=np.int32)
            cp.cumsum(img_counts, out=offsets[1:])
            offsets[0] = 0
            tot_imgs = int(offsets[n_pairs])
            img_idx = cp.empty(tot_imgs, dtype=np.int32)
            err = libpbc.sr_int3c2e_img_idx(
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_pair_sorting.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_pairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('sr_int3c2e_img_idx failed')
            log.debug1('ovlp nimgs=%d pairs=%d tot_imgs=%d. '
                       'double-lattice-sum: largest=%d, medium=%d',
                       nimgs_J, n_pairs, tot_imgs, img_counts[0], img_counts[n_pairs//2])
            t1 = log.timer_debug1('int3c2e_img_idx', *t0)

            # bas_ij stores the non-negligible primitive-pair indices.
            # p2c_mapping converts the bas_ij to contracted GTO-pair indices.
            I, i, J, j = cp.unravel_index(
                bas_ij, (bvk_ncells, nprimi, bvk_ncells, nprimj))
            i += ish0
            j += jsh0
            bas_ij = cp.ravel_multi_index(
                (I, i, J, j), (bvk_ncells, p_nbas, bvk_ncells, p_nbas))
            bas_ij = asarray(bas_ij, dtype=np.int32)
            ic = p2c_mapping[i] - c_shell_offsets[li]
            jc = p2c_mapping[j] - c_shell_offsets[lj]
            I %= bvk_ncells
            J %= bvk_ncells
            reduced_pair_idx = cp.ravel_multi_index(
                (I, ic, J, jc), (bvk_ncells, nctri, bvk_ncells, nctrj))
            bvk_nctri = bvk_ncells * nctri
            bvk_nctrj = bvk_ncells * nctrj
            c_pair_mask = cp.zeros(bvk_nctri*bvk_nctrj, dtype=bool)
            c_pair_mask[reduced_pair_idx] = True

            # c_pair_idx indicates the address of the **contracted** pair GTOS
            # within the (li,lj) sub-block. For each shell-pair, there are
            # nfij elements. Note, the nfij elements are sorted as [nfj,nfi]
            # (in F-order) while the shell indices within the c_pair_idx are
            # composed as i*nbas+j (in C-order). c_pair_idx points to the
            # address of the first element.
            c_pair_idx = cp.where(c_pair_mask)[0]
            n_ctr_pairs = len(c_pair_idx)

            # pair_mapping maps the primitive pair to the contracted pair
            pair_mapping_lookup = cp.empty(bvk_nctri*bvk_nctrj, dtype=np.int32)
            pair_mapping_lookup[c_pair_idx] = cp.arange(n_ctr_pairs)
            pair_mapping = asarray(pair_mapping_lookup[reduced_pair_idx], dtype=np.int32)
            log.timer_debug1(f'pair_mapping [{li},{lj}]', *t1)
            return img_idx, offsets, bas_ij, pair_mapping, c_pair_idx
        return gen_img_idx

    def make_img_idx_cache(self, cutoff=None):
        img_idx_cache = {}
        gen_img_idx = self.generate_img_idx(cutoff)
        l_counts = self.cell0_prim_l_counts
        lmax = len(l_counts) - 1
        ij_tasks = ((i, j) for i in range(lmax+1) for j in range(i+1))
        for li, lj in ij_tasks:
            if l_counts[li] == 0 or l_counts[lj] == 0:
                continue
            img_idx_cache[li, lj] = gen_img_idx(li, lj)
        return img_idx_cache

    def int3c2e_evaluator(self, verbose=None, img_idx_cache=None):
        log = logger.new_logger(self.cell, verbose)
        if self.int3c2e_envs is None:
            self.build(verbose)
        auxcell = self.sorted_auxcell
        bvkcell = self.bvkcell
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        aux_loc = auxcell.ao_loc
        naux = aux_loc[auxcell.nbas]
        _atm_cpu = self._atm_cpu
        _bas_cpu = self._bas_cpu
        _env_cpu = self._env_cpu

        l_counts = self.cell0_prim_l_counts
        p_shell_l_offsets = np.append(0, np.cumsum(l_counts))

        lmax = len(l_counts) - 1
        uniq_l = np.arange(lmax+1)
        nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
        kern = libpbc.fill_int3c2e

        if img_idx_cache is None:
            img_idx_cache = self.make_img_idx_cache()

        def evaluate_j3c(li, lj):
            if l_counts[li] == 0 or l_counts[lj] == 0:
                return cp.empty(0, dtype=np.int32), cp.empty((naux, 0))

            ish0, ish1 = p_shell_l_offsets[li:li+2]
            jsh0, jsh1 = p_shell_l_offsets[lj:lj+2]
            img_idx, img_offsets, bas_ij_idx, pair_mapping, c_pair_idx = img_idx_cache[li, lj]
            img_idx = asarray(img_idx)
            img_offsets = asarray(img_offsets)
            bas_ij_idx = asarray(bas_ij_idx)
            pair_mapping = asarray(pair_mapping)
            nfij = nfcart[li] * nfcart[lj]
            # Note the storage order for ij_pair: i takes the smaller stride.
            n_ctr_pairs = len(c_pair_idx)
            n_prim_pairs = len(bas_ij_idx)
            if n_prim_pairs == 0:
                return cp.empty(0, dtype=np.int32), cp.empty((naux, 0))

            # eri3c is sorted as (naux, nfj, nfi, n_ctr_pairs)
            eri3c = cp.zeros((naux, nfij*n_ctr_pairs))

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                k0 = aux_loc[ksh0]
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk)
                log.debug2(f'prim_pairs={n_prim_pairs} int3c2e_scheme for %s: %s', lll, scheme)
                err = kern(
                    ctypes.cast(eri3c[k0:].data.ptr, ctypes.c_void_p),
                    ctypes.byref(self.int3c2e_envs),
                    (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice),
                    ctypes.c_int(naux),
                    ctypes.c_int(n_prim_pairs),
                    ctypes.c_int(n_ctr_pairs),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(bvkcell.natm),
                    _bas_cpu.ctypes, ctypes.c_int(bvkcell.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
            return c_pair_idx, eri3c
        return evaluate_j3c

    def int3c2e_generator(self, verbose=None, img_idx_cache=None):
        log = logger.new_logger(self.cell, verbose)
        cput0 = log.init_timer()
        evaluate = self.int3c2e_evaluator(verbose, img_idx_cache)
        t1 = log.timer_debug1('initialize int3c2e_kernel', *cput0)
        timing_collection = {}
        kern_counts = 0

        lmax = len(self.cell0_prim_l_counts) - 1
        ij_tasks = ((i, j) for i in range(lmax+1) for j in range(i+1))
        for li, lj in ij_tasks:
            c_pair_idx, eri3c = evaluate(li, lj)
            if len(c_pair_idx) == 0:
                continue
            if log.verbose >= logger.DEBUG1:
                ll = f'{ANGULAR[li]}{ANGULAR[lj]}'
                t1, t1p = log.timer_debug1(f'processing {ll}, pairs={len(c_pair_idx)}', *t1), t1
                if ll not in timing_collection:
                    timing_collection[ll] = 0
                timing_collection[ll] += t1[1] - t1p[1]
                kern_counts += 1
            yield li, lj, c_pair_idx, eri3c

        if log.verbose >= logger.DEBUG1:
            log.timer('int3c2e', *cput0)
            for ll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', ll, t)

    def int3c2e_kernel(self, verbose=None, img_idx_cache=None):
        raise NotImplementedError(
            'The entire int3c2e tensor evaluated in one kernel is not supported')

def _conc_locs(ao_loc1, ao_loc2):
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return cp.array(comp_loc, dtype=np.int32)

def int3c2e_scheme(li, lj, lk, shm_size=SHM_SIZE):
    order = li + lj + lk
    nroots = (order//2 + 1) * 2

    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3 + nroots*2 + 7
    nksp_max = shm_size//(unit*8)
    nksp_max = _nearest_power2(nksp_max)

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    gout_size = nfi * nfj * nfk
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)

    # Align nksh*gout_stride to warp size
    if gout_stride < 32:
        nksh_per_block = 32 // gout_stride
        nsp_per_block = min(THREADS // 32, nksp_max // nksh_per_block)
    else:
        nksh_per_block = THREADS // gout_stride
        nsp_per_block = 1
    if nksp_max < nksh_per_block:
        raise RuntimeError('GOUT_WIDTH too small or not enough shared memory')

    gout_stride = THREADS // (nksh_per_block*nsp_per_block)
    return nksh_per_block, gout_stride, nsp_per_block

# This modified rcut estimation function will be available in pyscf-2.8 or newer
def estimate_rcut(cell, auxcell, omega):
    '''Estimate rcut for 3c2e SR-integrals'''
    if cell.nbas == 0 or auxcell.nbas == 0:
        return np.zeros(1)

    if omega == 0:
        # No SR integrals in int3c2e if omega=0
        assert cell.dimension == 0
        return np.zeros(1)

    precision = cell.precision
    ak, ck, lk = most_diffused_pgto(auxcell)

    # the most diffused orbital basis
    cell_exps, cs = extract_pgto_params(cell, 'diffused')
    ls = cell._bas[:,ANG_OF]
    r2_cell = np.log(cs**2 / precision * 10**ls) / cell_exps
    ai_idx = r2_cell.argmax()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    li = ls[ai_idx]
    lj = ls
    ci = cs[ai_idx]
    cj = cs

    aij = ai + aj
    lij = li + lj
    l3 = lij + lk
    theta = 1./(omega**-2 + 1./aij + 1./ak)
    norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
    c1 = ci * cj * ck * norm_ang
    sfac = aij*aj/(aij*aj + ai*theta)
    fl = 2
    fac = 2**li*np.pi**2.5*c1 * theta**(l3-.5)
    rad = cell.vol**(-1./3) * cell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*cell.rcut/(cell.vol*theta) + surface
    fac *= lattice_sum_factor
    fac /= aij**(li+1.5) * ak**(lk+1.5) * aj**lj
    fac *= fl / precision

    r0 = cell.rcut  # initial guess
    r0 = (np.log(fac * (sfac*r0+1e-200)**(l3-1) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * (sfac*r0+1e-200)**(l3-1) + 1.) / (sfac*theta))**.5
    rcut = r0
    return rcut

def _estimate_shl_pairs_per_block(li, lj, nshl_pair):
    return _nearest_power2(THREADS*25 // ((li+2)*(lj+2)), return_leq=False)
