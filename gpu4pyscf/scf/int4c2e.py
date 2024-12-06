# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import ctypes
import copy
import numpy as np
import scipy.linalg
import cupy
from pyscf import gto
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import block_c2s_diag, cart2sph, block_diag, contract, load_library, c2s_l
from gpu4pyscf.lib import logger

LMAX_ON_GPU = 4
FREE_CUPY_CACHE = True
BINSIZE = 128   # TODO bug for 256
libgvhf = load_library('libgvhf')
libgint = load_library('libgint')

_einsum = cupy.einsum
"""
def loop_int3c2e_general(intopt, ip_type='', omega=None, stream=None):
    '''
    loop over all int3c2e blocks
    - outer loop for k
    - inner loop for ij pair
    '''
    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    if ip_type == '':       order = 0
    if ip_type == 'ip1':    order = 1
    if ip_type == 'ip2':    order = 1
    if ip_type == 'ipip1':  order = 2
    if ip_type == 'ip1ip2': order = 2
    if ip_type == 'ipvip1': order = 2
    if ip_type == 'ipip2':  order = 2

    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()

    nao = intopt.mol.nao
    naux = intopt.auxmol.nao
    norb = nao + naux + 1

    comp = 3**order

    nbins = 1
    for aux_id, log_q_kl in enumerate(intopt.aux_log_qs):
        cp_kl_id = aux_id + len(intopt.log_qs)
        lk = intopt.aux_angular[aux_id]

        for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
            ni = i1 - i0
            nj = j1 - j0
            nk = k1 - k0

            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

            ao_offsets = np.array([i0,j0,nao+1+k0,nao], dtype=np.int32)
            strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

            int3c_blk = cupy.zeros([comp, nk, nj, ni], order='C', dtype=np.float64)
            err = fn(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError(f'GINT_fill_int3c2e general failed, err={err}')

            int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
            int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = intopt.sph_ao_loc[cpi], intopt.sph_ao_loc[cpi+1]
            j0, j1 = intopt.sph_ao_loc[cpj], intopt.sph_ao_loc[cpj+1]
            k0, k1 = intopt.sph_aux_loc[aux_id], intopt.sph_aux_loc[aux_id+1]

            yield i0,i1,j0,j1,k0,k1,int3c_blk


def get_int3c2e_ip(mol, auxmol=None, ip_type=1, auxbasis='weigend+etb', direct_scf_tol=1e-13, omega=None, stream=None):
    '''
    Generate full int3c2e_ip tensor on GPU
    ip_type == 1: int3c2e_ip1
    ip_type == 2: int3c2e_ip2
    '''
    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()
    if auxmol is None:
        from pyscf.df.addons import make_auxmol
        auxmol = make_auxmol(mol, auxbasis)

    nao_sph = mol.nao
    naux_sph = auxmol.nao

    intopt = _VHFOpt(mol, auxmol, 'int2e')
    intopt.build(direct_scf_tol, diag_block_with_triu=True)

    nao = intopt.mol.nao
    naux = intopt.auxmol.nao
    norb = nao + naux + 1

    int3c = cupy.zeros([3, naux_sph, nao_sph, nao_sph], order='C')
    nbins = 1
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]

        for aux_id, log_q_kl in enumerate(intopt.aux_log_qs):
            cp_kl_id = aux_id + len(intopt.log_qs)
            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
            ni = i1 - i0
            nj = j1 - j0
            nk = k1 - k0
            lk = intopt.aux_angular[aux_id]

            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

            ao_offsets = np.array([i0,j0,nao+1+k0,nao], dtype=np.int32)
            strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

            int3c_blk = cupy.zeros([3, nk, nj, ni], order='C', dtype=np.float64)
            err = fn(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError("int3c2e_ip failed\n")

            int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
            int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
            int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

            i0, i1 = intopt.sph_ao_loc[cpi], intopt.sph_ao_loc[cpi+1]
            j0, j1 = intopt.sph_ao_loc[cpj], intopt.sph_ao_loc[cpj+1]
            k0, k1 = intopt.sph_aux_loc[aux_id], intopt.sph_aux_loc[aux_id+1]

            int3c[:, k0:k1, j0:j1, i0:i1] = int3c_blk
    ao_idx = np.argsort(intopt.sph_ao_idx)
    aux_idx = np.argsort(intopt.sph_aux_idx)
    int3c = int3c[cupy.ix_(np.arange(3), aux_idx, ao_idx, ao_idx)]

    return int3c.transpose([0,3,2,1])
"""

def get_int4c2e(mol, vhfopt=None, direct_scf_tol=1e-13, aosym=True, omega=None, stream=None):
    '''
    Generate full int4c2e tensor on GPU
    '''

    if omega is None: omega = 0.0
    if vhfopt is None: vhfopt = _VHFOpt(mol, 'int2e').build(direct_scf_tol)
    if stream is None: stream = cupy.cuda.get_current_stream()

    nao = vhfopt.mol.nao
    norb = nao

    int4c = cupy.zeros([nao, nao, nao, nao], order='F')
    ao_offsets = np.array([0, 0, 0, 0], dtype=np.int32)
    strides = np.array([1, nao, nao*nao, nao*nao*nao], dtype=np.int32)
    for cp_ij_id, log_q_ij in enumerate(vhfopt.log_qs):
        for cp_kl_id, log_q_kl in enumerate(vhfopt.log_qs):
            bins_locs_ij = vhfopt.bins[cp_ij_id]
            bins_locs_kl = vhfopt.bins[cp_kl_id]
            nbins_locs_ij = len(bins_locs_ij) - 1
            nbins_locs_kl = len(bins_locs_kl) - 1
            bins_floor_ij = vhfopt.bins_floor[cp_ij_id]
            bins_floor_kl = vhfopt.bins_floor[cp_kl_id]
            log_cutoff = np.log(direct_scf_tol)
            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                vhfopt.bpcache,
                ctypes.cast(int4c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
                bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins_locs_ij),
                ctypes.c_int(nbins_locs_kl),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(log_cutoff),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError("int2c2e failed\n")

    coeff = vhfopt.coeff
    int4c = cupy.einsum('ijkl,ip->pjkl', int4c, coeff)
    int4c = cupy.einsum('pjkl,jq->pqkl', int4c, coeff)
    int4c = cupy.einsum('pqkl,kr->pqrl', int4c, coeff)
    int4c = cupy.einsum('pqrl,ls->pqrs', int4c, coeff)

    return int4c

def get_int4c2e_jk(mol, dm, vhfopt=None, direct_scf_tol=1e-13, with_k=True, omega=None, stream=None):

    if omega is None: omega = 0.0
    if vhfopt is None: vhfopt = _VHFOpt(mol, 'int2e').build(direct_scf_tol)
    if stream is None: stream = cupy.cuda.get_current_stream()

    coeff = cupy.asarray(vhfopt.coeff)
    dm_sorted = cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff)

    log_qs = vhfopt.log_qs
    ncptype = len(log_qs)
    cp_idx, cp_jdx = np.tril_indices(ncptype)
    l_ctr_shell_locs = vhfopt.l_ctr_offsets
    l_ctr_ao_locs = vhfopt.mol.ao_loc[l_ctr_shell_locs]

    nao = vhfopt.mol.nao
    norb = nao
    vj = cupy.zeros([nao, nao])
    vk = cupy.zeros([nao, nao])
    for cp_ij_id, log_q_ij in enumerate(vhfopt.log_qs):
        for cp_kl_id, log_q_kl in enumerate(vhfopt.log_qs[:cp_ij_id+1]):
            cpi = cp_idx[cp_ij_id]
            cpj = cp_jdx[cp_ij_id]
            cpk = cp_idx[cp_kl_id]
            cpl = cp_jdx[cp_kl_id]
            i0, i1 = l_ctr_ao_locs[cpi], l_ctr_ao_locs[cpi+1]
            j0, j1 = l_ctr_ao_locs[cpj], l_ctr_ao_locs[cpj+1]
            k0, k1 = l_ctr_ao_locs[cpk], l_ctr_ao_locs[cpk+1]
            l0, l1 = l_ctr_ao_locs[cpl], l_ctr_ao_locs[cpl+1]
            ni = i1 - i0
            nj = j1 - j0
            nk = k1 - k0
            nl = l1 - l0
            bins_locs_ij = vhfopt.bins[cp_ij_id]
            bins_locs_kl = vhfopt.bins[cp_kl_id]
            nbins_locs_ij = len(bins_locs_ij) - 1
            nbins_locs_kl = len(bins_locs_kl) - 1
            bins_floor_ij = vhfopt.bins_floor[cp_ij_id]
            bins_floor_kl = vhfopt.bins_floor[cp_kl_id]
            int4c = cupy.zeros([nl, nk, nj, ni], order='C')
            ao_offsets = np.array([i0, j0, k0, l0], dtype=np.int32)
            strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)
            log_cutoff = np.log(direct_scf_tol)

            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                vhfopt.bpcache,
                ctypes.cast(int4c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
                bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins_locs_ij),
                ctypes.c_int(nbins_locs_kl),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(log_cutoff),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError("int4c2e failed\n")

            if cp_ij_id == cp_kl_id: int4c *= 0.5
            contract('lkji,kl->ij', int4c, dm_sorted[k0:k1,l0:l1], alpha=1.0, beta=1.0, out=vj[i0:i1,j0:j1])
            contract('lkji,ij->kl', int4c, dm_sorted[i0:i1,j0:j1], alpha=1.0, beta=1.0, out=vj[k0:k1,l0:l1])

            contract('lkji,jl->ik', int4c, dm_sorted[j0:j1,l0:l1], alpha=1.0, beta=1.0, out=vk[i0:i1,k0:k1])
            contract('lkji,jk->il', int4c, dm_sorted[j0:j1,k0:k1], alpha=1.0, beta=1.0, out=vk[i0:i1,l0:l1])
            contract('lkji,il->jk', int4c, dm_sorted[i0:i1,l0:l1], alpha=1.0, beta=1.0, out=vk[j0:j1,k0:k1])
            contract('lkji,ik->jl', int4c, dm_sorted[i0:i1,k0:k1], alpha=1.0, beta=1.0, out=vk[j0:j1,l0:l1])

    vj = cupy.einsum('ip,ij,jq->pq', coeff, vj, coeff)
    vk = cupy.einsum('ip,ij,jq->pq', coeff, vk, coeff)
    vj = vj + vj.T
    vj *= 2.0
    vk = vk + vk.T
    return vj, vk

def get_int4c2e_ovov(mol, orbo, orbv, vhfopt=None, direct_scf_tol=1e-13, stream=None, omega=None):
    '''
    Generate 2-electron integrals (ov|ov) on GPU
    '''

    if omega is None: omega = 0.0
    if vhfopt is None: vhfopt = _VHFOpt(mol, 'int2e').build(direct_scf_tol)
    if stream is None: stream = cupy.cuda.get_current_stream()

    orbo = cupy.asarray(orbo)
    orbv = cupy.asarray(orbv)
    coeff = vhfopt.coeff

    orbo = coeff @ orbo
    orbv = coeff @ orbv

    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    ovov = cupy.zeros([nocc, nvir, nocc, nvir], order='C')
    for i0,i1,j0,j1,k0,k1,l0,l1,int4c in loop_int4c2e_general(vhfopt):
        int4c_oaaa = _einsum('lkji,io->ojkl', int4c[0], orbo[i0:i1])

        int4c_oaoa = _einsum('ojkl,kr->ojrl', int4c_oaaa, orbo[k0:k1])
        int4c_ovoa = _einsum('ojrl,jp->oprl', int4c_oaoa, orbv[j0:j1])
        ovov += _einsum('oprl,lq->oprq', int4c_ovoa, orbv[l0:l1])

        int4c_oaao = _einsum('ojkl,lr->ojkr', int4c_oaaa, orbo[l0:l1])
        int4c_ovao = _einsum('ojkr,jp->opkr', int4c_oaao, orbv[j0:j1])
        ovov += _einsum('opkr,kq->oprq', int4c_ovao, orbv[k0:k1])

        int4c_aoaa = _einsum('lkji,jo->iokl', int4c[0], orbo[j0:j1])

        int4c_aooa = _einsum('iokl,kr->iorl', int4c_aoaa, orbo[k0:k1])
        int4c_vooa = _einsum('iorl,ip->porl', int4c_aooa, orbv[i0:i1])
        ovov += _einsum('porl,lq->oprq', int4c_vooa, orbv[l0:l1])

        int4c_aoao = _einsum('iokl,lr->iokr', int4c_aoaa, orbo[l0:l1])
        int4c_voao = _einsum('iokr,ip->pokr', int4c_aoao, orbv[i0:i1])
        ovov += _einsum('pokr,kq->oprq', int4c_voao, orbv[k0:k1])

    ovov += ovov.transpose([2,3,0,1])
    return ovov

def loop_int4c2e_general(intopt, ip_type='', direct_scf_tol=1e-13, omega=None, stream=None):
    '''
    loop over all int2e blocks
    - outer loop for ij pair
    - inner loop for kl pair
    '''
    if ip_type == '':
        order = 0
        fn = getattr(libgint, 'GINTfill_int2e')
    else:
        fn = getattr(libgint, 'GINTfill_int2e_' + ip_type)
    if ip_type == 'ip1':    order = 1
    if ip_type == 'ip2':    order = 1
    if ip_type == 'ipip1':  order = 2
    if ip_type == 'ip1ip2': order = 2
    if ip_type == 'ipvip1': order = 2
    if ip_type == 'ipip2':  order = 2

    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()

    comp = 3**order
    nao = intopt.mol.nao
    log_qs = intopt.log_qs
    ncptype = len(log_qs)
    cp_idx, cp_jdx = np.tril_indices(ncptype)
    cart_ao_loc = intopt.mol.ao_loc_nr(cart=True)
    cart_ao_loc = [cart_ao_loc[cp] for cp in intopt.l_ctr_offsets]

    log_cutoff = np.log(direct_scf_tol)
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        i0, i1 = cart_ao_loc[cpi], cart_ao_loc[cpi+1]
        j0, j1 = cart_ao_loc[cpj], cart_ao_loc[cpj+1]
        ni = i1 - i0
        nj = j1 - j0
        for cp_kl_id, log_q_kl in enumerate(log_qs[:cp_ij_id+1]):
            cpk = cp_idx[cp_kl_id]
            cpl = cp_jdx[cp_kl_id]
            k0, k1 = cart_ao_loc[cpk], cart_ao_loc[cpk+1]
            l0, l1 = cart_ao_loc[cpl], cart_ao_loc[cpl+1]
            nk = k1 - k0
            nl = l1 - l0
            bins_locs_ij = intopt.bins[cp_ij_id]
            bins_locs_kl = intopt.bins[cp_kl_id]
            nbins_locs_ij = len(bins_locs_ij) - 1
            nbins_locs_kl = len(bins_locs_kl) - 1
            bins_floor_ij = intopt.bins_floor[cp_ij_id]
            bins_floor_kl = intopt.bins_floor[cp_kl_id]

            ao_offsets = np.array([i0,j0,k0,l0], dtype=np.int32)
            strides = np.array([1,ni,ni*nj,ni*nj*nk], dtype=np.int32)

            int4c = cupy.zeros([comp,nl,nk,nj,ni], order='C', dtype=np.float64)
            err = fn(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int4c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
                bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins_locs_ij),
                ctypes.c_int(nbins_locs_kl),
                ctypes.c_int(cp_ij_id),
                ctypes.c_int(cp_kl_id),
                ctypes.c_double(log_cutoff),
                ctypes.c_double(omega))

            if err != 0:
                raise RuntimeError(f'GINT_fill_int4c2e general failed, err={err}')
            if cp_ij_id == cp_kl_id:
                int4c *= 0.5
            yield i0,i1,j0,j1,k0,k1,l0,l1,int4c

class _VHFOpt:
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    def __init__(self, mol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol, self.coeff = basis_seg_contraction(mol)
        self.coeff = cupy.asarray(self.coeff)
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

    def build(self, cutoff=1e-13, group_size=None, diag_block_with_triu=False):
        mol = self.mol
        cput0 = logger.init_timer(mol)
        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = mol._bas[:,[gto.ANG_OF, gto.NPRIM_OF]]
        uniq_l_ctr, _, inv_idx, l_ctr_counts = np.unique(
            l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)

        # Limit the number of AOs in each group
        if group_size is not None:
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
                uniq_l_ctr, l_ctr_counts, group_size)

        if mol.verbose >= logger.DEBUG1:
            logger.debug1(mol, 'Number of shells for each [l, nprim] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug1(mol, '    %s : %s', l_ctr, n)

        sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
        # Sort contraction coefficients before updating self.mol
        ao_loc = mol.ao_loc_nr(cart=True)
        nao = ao_loc[-1]
        # Some addressing problems in GPU kernel code
        assert nao < 32768
        ao_idx = np.array_split(np.arange(nao), ao_loc[1:-1])
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        self.coeff = self.coeff[ao_idx]
        # Sort basis inplace
        mol._bas = mol._bas[sorted_idx]

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax >= LMAX_ON_GPU:
            self.g_shls = l_slices[LMAX_ON_GPU:LMAX_ON_GPU+2].tolist()
        else:
            self.g_shls = []
        if lmax > LMAX_ON_GPU:
            self.h_shls = l_slices[LMAX_ON_GPU+1:].tolist()
        else:
            self.h_shls = []

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        q_cond = self.get_q_cond()
        cput1 = logger.timer(mol, 'Initialize q_cond', *cput0)
        log_qs = []
        pair2bra = []
        pair2ket = []
        bins = []
        bins_floor = []
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        for i, (p0, p1) in enumerate(zip(l_ctr_offsets[:-1], l_ctr_offsets[1:])):
            if uniq_l_ctr[i,0] > LMAX_ON_GPU:
                # no integrals with h functions should be evaluated on GPU
                continue

            for q0, q1 in zip(l_ctr_offsets[:i], l_ctr_offsets[1:i+1]):
                q_sub = q_cond[p0:p1,q0:q1]
                idx = np.argwhere(q_sub > cutoff)
                q_sub = q_sub[idx[:,0], idx[:,1]]
                log_q = np.log(q_sub)
                log_q[log_q > 0] = 0
                nbins = (len(log_q) + BINSIZE)//BINSIZE
                s_index, bin_floor = _make_s_index(log_q, nbins=nbins, cutoff=cutoff)

                ishs = idx[:,0]
                jshs = idx[:,1]
                idx = np.lexsort((ishs, jshs, s_index), axis=-1)
                ishs = ishs[idx]
                jshs = jshs[idx]
                s_index = s_index[idx]

                ishs += p0
                jshs += q0
                pair2bra.append(ishs)
                pair2ket.append(jshs)
                bins.append(_make_bins(s_index, nbins=nbins))
                bins_floor.append(bin_floor)
                log_qs.append(cupy.asarray(log_q[idx]))

            q_sub = q_cond[p0:p1,p0:p1]
            idx = np.argwhere(q_sub > cutoff)
            if not diag_block_with_triu:
                # Drop the shell pairs in the upper triangle for diagonal blocks
                mask = idx[:,0] >= idx[:,1]
                idx = idx[mask,:]

            q_sub = q_sub[idx[:,0], idx[:,1]]
            log_q = np.log(q_sub)
            log_q[log_q > 0] = 0
            nbins = (len(log_q) + BINSIZE)//BINSIZE
            s_index, bin_floor = _make_s_index(log_q, nbins=nbins, cutoff=cutoff)
            ishs = idx[:,0]
            jshs = idx[:,1]
            idx = np.lexsort((ishs, jshs, s_index), axis=-1)
            ishs = ishs[idx]
            jshs = jshs[idx]
            s_index = s_index[idx]

            ishs += p0
            jshs += p0
            pair2bra.append(ishs)
            pair2ket.append(jshs)
            bins.append(_make_bins(s_index, nbins=nbins))
            bins_floor.append(bin_floor)
            log_qs.append(cupy.asarray(log_q[idx]))

        # TODO
        self.pair2bra = pair2bra
        self.pair2ket = pair2ket
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = l_ctr_offsets
        self.bas_pair2shls = np.hstack(
            pair2bra + pair2ket).astype(np.int32).reshape(2,-1)

        self.bas_pairs_locs = np.append(
            0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        self.bins = bins
        self.bins_floor = bins_floor
        self.log_qs = log_qs
        ao_loc = mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)
        self.bpcache = ctypes.POINTER(BasisProdCache)()
        if diag_block_with_triu:
            scale_shellpair_diag = 1.
        else:
            scale_shellpair_diag = 0.5
        libgvhf.GINTinit_basis_prod(
            ctypes.byref(self.bpcache), ctypes.c_double(scale_shellpair_diag),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            self.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            self.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p))
        logger.timer(mol, 'Initialize GPU cache', *cput1)
        return self

    init_cvhf_direct = _vhf.VHFOpt.init_cvhf_direct
    get_q_cond       = _vhf.VHFOpt.get_q_cond
    set_dm           = _vhf.VHFOpt.set_dm

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        libgvhf.GINTdel_basis_prod(ctypes.byref(self.bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

class BasisProdCache(ctypes.Structure):
    pass

def basis_seg_contraction(mol, allow_replica=False):
    '''transform generally contracted basis to segment contracted basis
    Kwargs:
        allow_replica:
            transform the generally contracted basis to replicated
            segment-contracted basis
    '''
    bas_templates = {}
    _bas = []
    _env = mol._env.copy()
    contr_coeff = []
    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        key = tuple(mol._bas[ib0:ib1,gto.PTR_EXP])
        if key in bas_templates:
            bas_of_ia, coeff = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,gto.ATOM_OF] = ia
        else:
            # Generate the template for decontracted basis
            coeff = []
            bas_of_ia = []
            for shell in mol._bas[ib0:ib1]:
                l = shell[gto.ANG_OF]
                nf = (l + 1) * (l + 2) // 2
                nctr = shell[gto.NCTR_OF]
                if nctr == 1:
                    bas_of_ia.append(shell)
                    coeff.append(np.eye(nf))
                    continue
                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[gto.NPRIM_OF]
                pcoeff = shell[gto.PTR_COEFF]
                if allow_replica:
                    coeff.extend([np.eye(nf)] * nctr)
                    bs = np.repeat(shell[np.newaxis], nctr, axis=0)
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr, nprim)
                    bas_of_ia.append(bs)
                else:
                    pexp = shell[gto.PTR_EXP]
                    exps = _env[pexp:pexp+nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    c = _env[pcoeff:pcoeff+nprim*nctr].reshape(nctr,nprim)
                    c = np.einsum('ip,p,ef->iepf', c, 1/norm, np.eye(nf))
                    coeff.append(c.reshape(nf*nctr, nf*nprim).T)

                    _env[pcoeff:pcoeff+nprim] = norm
                    bs = np.repeat(shell[np.newaxis], nprim, axis=0)
                    bs[:,gto.NPRIM_OF] = 1
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_EXP] = np.arange(pexp, pexp+nprim)
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim)
                    bas_of_ia.append(bs)

            bas_of_ia = np.vstack(bas_of_ia)
            bas_templates[key] = (bas_of_ia, coeff)

        _bas.append(bas_of_ia)
        contr_coeff.extend(coeff)

    pmol = mol.copy()
    pmol.cart = True
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env
    contr_coeff = scipy.linalg.block_diag(*contr_coeff)

    if not mol.cart:
        contr_coeff = contr_coeff.dot(mol.cart2sph_coeff())
    return pmol, contr_coeff

def _make_s_index_offsets(log_q, nbins=10, cutoff=1e-12):
    '''Divides the shell pairs to "nbins" collections down to "cutoff"'''
    scale = nbins / np.log(min(cutoff, .1))
    s_index = np.floor(scale * log_q).astype(np.int32)
    bins = np.bincount(s_index)
    if bins.size < nbins:
        bins = np.append(bins, np.zeros(nbins-bins.size, dtype=np.int32))
    else:
        bins = bins[:nbins]
    assert bins.max() < 65536 * 8
    return np.append(0, np.cumsum(bins)).astype(np.int32)

def _make_s_index(log_q, nbins=10, cutoff=1e-12):
    '''Divides the shell pairs to "nbins" collections down to "cutoff"'''
    scale = nbins / np.log(min(cutoff, .1))
    s_index = np.floor(scale * log_q).astype(np.int32)
    bins_floor = np.arange(nbins) / scale
    return s_index, bins_floor

def _make_bins(s_index, nbins=10):
    bins = np.bincount(s_index)
    if bins.size < nbins:
        bins = np.append(bins, np.zeros(nbins-bins.size, dtype=np.int32))
    else:
        bins = bins[:nbins]
    assert bins.max() < 65536 * 8
    return np.append(0, np.cumsum(bins)).astype(np.int32)

def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size):
    '''Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
    l = uniq_l_ctr[:,0]
    _l_ctrs = []
    _l_ctr_counts = []
    for l_ctr, counts in zip(uniq_l_ctr, l_ctr_counts):
        l = l_ctr[0]
        nf = (l + 1) * (l + 2) // 2
        max_shells = max(group_size // nf, 2)
        if l > LMAX_ON_GPU or counts <= max_shells:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(counts)
            continue

        nsubs, rests = counts.__divmod__(max_shells)
        _l_ctrs.extend([l_ctr] * nsubs)
        _l_ctr_counts.extend([max_shells] * nsubs)
        if rests > 0:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(rests)
    uniq_l_ctr = np.vstack(_l_ctrs)
    l_ctr_counts = np.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts
