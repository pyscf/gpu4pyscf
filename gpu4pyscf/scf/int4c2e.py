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
import cupy
from pyscf.scf import _vhf
from gpu4pyscf.scf.hf import BasisProdCache, _make_s_index_offsets, _VHFOpt
from gpu4pyscf.lib.cupy_helper import block_c2s_diag, cart2sph, block_diag, contract, load_library, c2s_l

libgint = load_library('libgint')
libcupy_helper = load_library('libcupy_helper')

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
    from gpu4pyscf.scf.hf import _VHFOpt
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