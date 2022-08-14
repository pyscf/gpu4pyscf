# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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
import contextlib
import numpy as np
import cupy
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.dft.rks import KohnShamDFT
from gpu4pyscf.scf.hf import basis_seg_contraction
from gpu4pyscf.lib.utils import patch_cpu_kernel

LMAX_ON_GPU = 4
ALIGNED = 128
# Should we release the cupy cache?
FREE_CUPY_CACHE = True

libgdft = lib.load_library('libgdft')
libgdft.GDFTeval_gto.restype = ctypes.c_int

def eval_ao(ni, mol, coords, deriv=0, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    assert shls_slice is None
    ngrids = coords.shape[0]
    coords = np.asarray(coords.T, order='C')
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = _GDFTOpt(mol)
        # mol may be different to _GDFTOpt.mol.
        # nao should be consistent with the _GDFTOpt.mol object
        nao = opt.coeff.shape[0]
        ao = cupy.empty((comp, nao, ngrids))
        with opt.gdft_envs_cache():
            err = libgdft.GDFTeval_gto(
                ctypes.cast(ao.data.ptr, ctypes.c_void_p),
                ctypes.c_int(deriv), ctypes.c_int(opt.mol.cart),
                coords.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrids),
                opt.l_ctr_offsets.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(opt.l_ctr_offsets.size - 1),
                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p))
    else:
        nao = opt.coeff.shape[0]
        ao = cupy.empty((comp, nao, ngrids))
        err = libgdft.GDFTeval_gto(
            ctypes.cast(ao.data.ptr, ctypes.c_void_p),
            ctypes.c_int(deriv), ctypes.c_int(mol.cart),
            coords.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrids),
            opt.l_ctr_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(opt.l_ctr_offsets.size - 1),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('CUDA Error')

    ao = ao.transpose(0, 2, 1)
    if deriv == 0:
        ao = ao[0]
    return ao

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    dm = cupy.asarray(dm)
    if xctype == 'LDA' or xctype == 'HF':
        c0 = dm.dot(ao.T).T
        rho = _contract_rho(c0, ao)
    elif xctype in ('GGA', 'NLC'):
        rho = cupy.empty((4,ngrids))
        c0 = dm.dot(ao[0].T).T
        rho[0] = _contract_rho(c0, ao[0])
        for i in range(1, 4):
            rho[i] = _contract_rho(c0, ao[i])
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            c0 = dm.T.dot(ao[0].T).T
            for i in range(1, 4):
                rho[i] += _contract_rho(ao[i], c0)
    else:  # meta-GGA
        if with_lapl:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = cupy.empty((6,ngrids))
            tau_idx = 5
        else:
            rho = cupy.empty((5,ngrids))
            tau_idx = 4
        c0 = dm.dot(ao[0].T).T
        rho[0] = _contract_rho(c0, ao[0])

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = dm.dot(ao[i].T).T
            rho[tau_idx] += _contract_rho(c1, ao[i])
            rho[i] = _contract_rho(c0, ao[i])
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho(c1, ao[0])
        rho[tau_idx] *= .5  # tau = 1/2 (\nabla f)^2
    return rho.get()

@patch_cpu_kernel(numint.nr_rks)
def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = ni.gdftopt = _GDFTOpt(mol)

    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    dms = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff)
           for dm in dms.reshape(-1,nao0,nao0)]
    nset = len(dms)

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = cupy.zeros((nset, nao, nao))

    with opt.gdft_envs_cache():
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        if xctype == 'LDA':
            block_size = int((mem_avail*.7/8/3/nao - nao*2)/ ALIGNED) * ALIGNED
        else:
            block_size = int((mem_avail*.7/8/8/nao - nao*2)/ ALIGNED) * ALIGNED
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)
        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids = grids.weights.size
        for p0, p1 in lib.prange(0, ngrids, block_size):
            ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], ao_deriv)
            weight = grids.weights[p0:p1]
            for i in range(nset):
                rho = eval_rho(opt.mol, ao, dms[i], xctype=xctype, hermi=1, with_lapl=False)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
                if xctype == 'LDA':
                    den = rho * weight
                    wv = weight * vxc[0]
                    vmat[i] += ao.T.dot(_scale_ao(ao, wv))
                elif xctype == 'GGA':
                    den = rho[0] * weight
                    wv = vxc * weight
                    wv[0] *= .5
                    vmat[i] += ao[0].T.dot(_scale_ao(ao, wv))
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                elif xctype == 'MGGA':
                    den = rho[0] * weight
                    wv = vxc * weight
                    wv[[0, 4]] *= .5  # *.5 for v+v.T
                    vmat[i] += ao[0].T.dot(_scale_ao(ao[:4], wv[:4]))
                    vmat[i] += _tau_dot(ao, ao, wv[4])
                elif xctype == 'HF':
                    pass
                else:
                    raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
            ao = None

    vmat = [cupy.einsum('pi,pq,qj->ij', coeff, v, coeff).get() for v in vmat]
    if xctype != 'LDA':
        for i in range(nset):
            lib.transpose_sum(vmat[i], inplace=True)

    if FREE_CUPY_CACHE:
        dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, np.asarray(vmat)


@patch_cpu_kernel(numint.nr_uks)
def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = ni.gdftopt = _GDFTOpt(mol)

    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dma, dmb = dms
    dm_shape = dma.shape
    dma = cupy.asarray(dma).reshape(-1,nao0,nao0)
    dmb = cupy.asarray(dmb).reshape(-1,nao0,nao0)
    dma = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff) for dm in dma]
    dmb = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff) for dm in dmb]
    nset = len(dma)

    nelec = np.zeros((2,nset))
    excsum = np.zeros(nset)
    vmata = cupy.zeros((nset, nao, nao))
    vmatb = cupy.zeros((nset, nao, nao))

    with opt.gdft_envs_cache():
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        if xctype == 'LDA':
            block_size = int((mem_avail*.7/8/3/nao - nao*2)/ ALIGNED) * ALIGNED
        else:
            block_size = int((mem_avail*.7/8/8/nao - nao*2)/ ALIGNED) * ALIGNED
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)
        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids = grids.weights.size
        for p0, p1 in lib.prange(0, ngrids, block_size):
            ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], ao_deriv)
            weight = grids.weights[p0:p1]
            for i in range(nset):
                rho_a = eval_rho(opt.mol, ao, dma[i], xctype=xctype, hermi=1)
                rho_b = eval_rho(opt.mol, ao, dmb[i], xctype=xctype, hermi=1)
                rho = (rho_a, rho_b)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
                if xctype == 'LDA':
                    den_a = rho_a * weight
                    den_b = rho_b * weight
                    wv = vxc[:,0] * weight
                    vmata[i] += ao.T.dot(_scale_ao(ao, wv[0]))
                    vmatb[i] += ao.T.dot(_scale_ao(ao, wv[1]))
                elif xctype == 'GGA':
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                    wv = vxc * weight
                    wv[:,0] *= .5
                    vmata[i] += ao[0].T.dot(_scale_ao(ao, wv[0]))
                    vmatb[i] += ao[0].T.dot(_scale_ao(ao, wv[1]))
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                elif xctype == 'MGGA':
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                    wv = vxc * weight
                    wv[:,[0, 4]] *= .5
                    vmata[i] += ao[0].T.dot(_scale_ao(ao[:4], wv[0,:4]))
                    vmatb[i] += ao[0].T.dot(_scale_ao(ao[:4], wv[1,:4]))
                    vmata[i] += _tau_dot(ao, ao, wv[0,4])
                    vmatb[i] += _tau_dot(ao, ao, wv[1,4])
                elif xctype == 'HF':
                    pass
                else:
                    raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')
                nelec[0,i] += den_a.sum()
                nelec[1,i] += den_b.sum()
                excsum[i] += np.dot(den_a, exc)
                excsum[i] += np.dot(den_b, exc)
            ao = None

    vmata = [cupy.einsum('pi,pq,qj->ij', coeff, v, coeff).get() for v in vmata]
    vmatb = [cupy.einsum('pi,pq,qj->ij', coeff, v, coeff).get() for v in vmatb]
    if xctype != 'LDA':
        for i in range(nset):
            lib.transpose_sum(vmata[i], inplace=True)
            lib.transpose_sum(vmatb[i], inplace=True)

    if FREE_CUPY_CACHE:
        dma = dmb = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        nelec = nelec.reshape(2)
        excsum = excsum[0]
        vmata = vmata[0]
        vmatb = vmatb[0]
    vmat = np.asarray([vmata, vmatb])
    return nelec, excsum, vmat


@patch_cpu_kernel(numint.get_rho)
def get_rho(ni, mol, dm, grids, max_memory=2000):
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = ni.gdftopt = _GDFTOpt(mol)

    coeff = cupy.asarray(opt.coeff)
    nao = coeff.shape[0]
    dm = cupy.einsum('pi,ij,qj->pq', coeff, cupy.asarray(dm), coeff)

    with opt.gdft_envs_cache():
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        block_size = int((mem_avail*.7/8/3/nao - nao*2)/ ALIGNED) * ALIGNED
        logger.debug1(mol, 'Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)
        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        ngrids = grids.weights.size
        rho = np.empty(ngrids)
        for p0, p1 in lib.prange(0, ngrids, block_size):
            ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], deriv=0)
            rho[p0:p1] = eval_rho(opt.mol, ao, dm, xctype='LDA', hermi=1)
            ao = None

    if FREE_CUPY_CACHE:
        dm = None
        cupy.get_default_memory_pool().free_all_blocks()
    return rho


@patch_cpu_kernel(numint.nr_rks_fxc)
def nr_rks_fxc(ni, mol, grids, xc_code, dm0=None, dms=None, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = ni.gdftopt = _GDFTOpt(mol)

    coeff = opt.coeff
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    dms = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff)
           for dm in dms.reshape(-1,nao0,nao0)]
    nset = len(dms)
    vmat = cupy.zeros((nset, nao, nao))

    with opt.gdft_envs_cache():
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        if xctype == 'LDA':
            block_size = int((mem_avail*.7/8/3/nao - nao*2)/ ALIGNED) * ALIGNED
        else:
            block_size = int((mem_avail*.7/8/8/nao - nao*2)/ ALIGNED) * ALIGNED
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)
        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids = grids.weights.size
        for p0, p1 in lib.prange(0, ngrids, block_size):
            ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], ao_deriv)
            weight = grids.weights[p0:p1]
            for i in range(nset):
                rho1 = eval_rho(opt.mol, ao, dms[i], xctype=xctype, hermi=hermi, with_lapl=False)
                if xctype == 'LDA':
                    wv = rho1 * fxc[0,0,p0:p1] * weight
                    vmat[i] += ao.T.dot(_scale_ao(ao, wv))
                elif xctype == 'GGA':
                    wv = np.einsum('xg,xyg->yg', rho1, fxc[:,:,p0:p1]) * weight
                    wv[0] *= .5
                    vmat[i] += ao[0].T.dot(_scale_ao(ao, wv))
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                else:
                    wv = np.einsum('xg,xyg->yg', rho1, fxc[:,:,p0:p1]) * weight
                    wv[[0, 4]] *= .5
                    vmat[i] += ao[0].T.dot(_scale_ao(ao[:4], wv[:4]))
                    vmat[i] += _tau_dot(ao, ao, wv[4])
            ao = None

    vmat = [cupy.einsum('pi,pq,qj->ij', coeff, v, coeff).get() for v in vmat]
    if xctype != 'LDA':
        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        for i in range(nset):
            lib.transpose_sum(vmat[i], inplace=True)

    if FREE_CUPY_CACHE:
        dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        vmat = vmat[0]
    return np.asarray(vmat)


@patch_cpu_kernel(numint.nr_rks_fxc_st)
def nr_rks_fxc_st(ni, mol, grids, xc_code, dm0=None, dms_alpha=None,
                  relativity=0, singlet=True, rho0=None, vxc=None, fxc=None,
                  max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    if singlet:
        fxc = fxc[0,:,0] + fxc[0,:,1]
    else:
        fxc = fxc[0,:,0] - fxc[0,:,1]
    return nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms_alpha, hermi=0, fxc=fxc)


@patch_cpu_kernel(numint.nr_uks_fxc)
def nr_uks_fxc(ni, mol, grids, xc_code, dm0=None, dms=None, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = ni.gdftopt = _GDFTOpt(mol)

    coeff = opt.coeff
    nao, nao0 = coeff.shape
    dma, dmb = dms
    dm_shape = dma.shape
    dma = cupy.asarray(dma).reshape(-1,nao0,nao0)
    dmb = cupy.asarray(dmb).reshape(-1,nao0,nao0)
    dma = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff) for dm in dma]
    dmb = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff) for dm in dmb]
    nset = len(dma)
    vmata = cupy.zeros((nset, nao, nao))
    vmatb = cupy.zeros((nset, nao, nao))

    with opt.gdft_envs_cache():
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        if xctype == 'LDA':
            block_size = int((mem_avail*.7/8/3/nao - nao*2)/ ALIGNED) * ALIGNED
        else:
            block_size = int((mem_avail*.7/8/8/nao - nao*2)/ ALIGNED) * ALIGNED
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)
        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids = grids.weights.size
        for p0, p1 in lib.prange(0, ngrids, block_size):
            ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], ao_deriv)
            weight = grids.weights[p0:p1]
            for i in range(nset):
                rho1a = eval_rho(opt.mol, ao, dma[i], xctype=xctype, hermi=hermi, with_lapl=False)
                rho1b = eval_rho(opt.mol, ao, dmb[i], xctype=xctype, hermi=hermi, with_lapl=False)
                rho1 = np.asarray([rho1a, rho1b])
                if xctype == 'LDA':
                    wv = np.einsum('ag,abg->bg', rho1, fxc[:,0,:,0,p0:p1]) * weight
                    vmata[i] += ao.T.dot(_scale_ao(ao, wv[0]))
                    vmatb[i] += ao.T.dot(_scale_ao(ao, wv[1]))
                elif xctype == 'GGA':
                    wv = np.einsum('axg,axbyg->byg', rho1, fxc[:,:,:,:,p0:p1]) * weight
                    wv[:,0] *= .5
                    vmata[i] += ao[0].T.dot(_scale_ao(ao, wv[0]))
                    vmatb[i] += ao[0].T.dot(_scale_ao(ao, wv[1]))
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                else:
                    wv = np.einsum('axg,axbyg->byg', rho1, fxc[:,:,:,:,p0:p1]) * weight
                    wv[:,[0, 4]] *= .5
                    vmata[i] += ao[0].T.dot(_scale_ao(ao[:4], wv[0,:4]))
                    vmatb[i] += ao[0].T.dot(_scale_ao(ao[:4], wv[1,:4]))
                    vmata[i] += _tau_dot(ao, ao, wv[0,4])
                    vmatb[i] += _tau_dot(ao, ao, wv[1,4])
            ao = None

    vmata = [cupy.einsum('pi,pq,qj->ij', coeff, v, coeff).get() for v in vmata]
    vmatb = [cupy.einsum('pi,pq,qj->ij', coeff, v, coeff).get() for v in vmatb]
    if xctype != 'LDA':
        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        for i in range(nset):
            lib.transpose_sum(vmata[i], inplace=True)
            lib.transpose_sum(vmatb[i], inplace=True)

    if FREE_CUPY_CACHE:
        dma = dmb = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        vmata = vmata[0]
        vmatb = vmatb[0]
    vmat = np.asarray([vmata, vmatb])
    return vmat


@patch_cpu_kernel(numint.cache_xc_kernel)
def cache_xc_kernel(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    max_memory=2000):
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 1
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        ao_deriv = 0

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        opt = ni.gdftopt = _GDFTOpt(mol)

    ngrids = grids.weights.size
    nao = opt.coeff.shape[0]

    def make_rdm1(mo_coeff, mo_occ):
        orbo = opt.coeff.dot(mo_coeff[:,mo_occ>0])
        dm = (orbo*mo_occ[mo_occ>0]).dot(orbo.T)
        return dm

    with opt.gdft_envs_cache():
        mem_avail = cupy.cuda.runtime.memGetInfo()[0]
        block_size = int((mem_avail*.7/8/3/nao - nao*2)/ ALIGNED) * ALIGNED
        logger.debug1(mol, 'Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)
        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if spin == 0:
            dm = make_rdm1(mo_coeff, mo_occ)
            rho = []
            for p0, p1 in lib.prange(0, ngrids, block_size):
                ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], ao_deriv)
                rho.append(eval_rho(opt.mol, ao, dm, xctype=xctype, hermi=1))
                ao = None
            rho = np.hstack(rho)
        else:
            dma = make_rdm1(mo_coeff[0], mo_occ[0])
            dmb = make_rdm1(mo_coeff[1], mo_occ[1])
            rhoa = []
            rhob = []
            for p0, p1 in lib.prange(0, ngrids, block_size):
                ao = eval_ao(ni, opt.mol, grids.coords[p0:p1], ao_deriv)
                rhoa.append(eval_rho(opt.mol, ao, dma, xctype=xctype, hermi=1))
                rhob.append(eval_rho(opt.mol, ao, dmb, xctype=xctype, hermi=1))
                ao = None
            rho = (np.hstack(rhoa), np.hstack(rhob))

    if FREE_CUPY_CACHE:
        dm = dma = dmb = None
        cupy.get_default_memory_pool().free_all_blocks()

    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
    return rho, vxc, fxc


class NumInt(numint.NumInt):
    device = 'gpu'
    def __init__(self):
        super().__init__()
        self.gdftopt = None

    get_rho = get_rho
    nr_rks = nr_rks
    nr_uks = nr_uks
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    cache_xc_kernel = cache_xc_kernel

def _contract_rho(bra, ket):
    if bra.flags.f_contiguous and ket.flags.f_contiguous:
        assert bra.shape == ket.shape
        ngrids, nao = bra.shape
        rho = cupy.empty(ngrids)
        err = libgdft.GDFTcontract_rho(
            ctypes.cast(rho.data.ptr, ctypes.c_void_p),
            ctypes.cast(bra.data.ptr, ctypes.c_void_p),
            ctypes.cast(ket.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids), ctypes.c_int(nao))
        if err != 0:
            raise RuntimeError('CUDA Error')
    else:
        rho = cupy.einsum('gi,gi->g', bra, ket)
    return rho

def _scale_ao(ao, wv):
    if wv.ndim == 1:
        if not ao.flags.f_contiguous:
            return cupy.einsum('pi,p->ip', ao, wv).T
        nvar = 1
        ngrids, nao = ao.shape
        assert wv.size == ngrids
    else:
        if not ao[0].flags.f_contiguous:
            return cupy.einsum('npi,np->ip', ao, wv).T
        nvar, ngrids, nao = ao.shape
        assert wv.shape == (nvar, ngrids)

    wv = cupy.asarray(wv)
    aow = cupy.empty((ngrids, nao), order='F')
    err = libgdft.GDFTscale_ao(
        ctypes.cast(aow.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao.data.ptr, ctypes.c_void_p),
        ctypes.cast(wv.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids), ctypes.c_int(nao), ctypes.c_int(nvar))
    if err != 0:
        raise RuntimeError('CUDA Error')
    return aow

def _tau_dot(bra, ket, wv):
    '''1/2 <nabla i| v | nabla j>'''
    wv = cupy.asarray(.5 * wv)
    mat  = bra[1].T.dot(_scale_ao(ket[1], wv))
    mat += bra[2].T.dot(_scale_ao(ket[2], wv))
    mat += bra[3].T.dot(_scale_ao(ket[3], wv))
    return mat

class _GDFTOpt:
    def __init__(self, mol):
        self.envs_cache = ctypes.POINTER(_GDFTEnvsCache)()
        pmol, coeff = basis_seg_contraction(mol, allow_replica=True)

        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = pmol._bas[:,[gto.ANG_OF, gto.NPRIM_OF]]
        uniq_l_ctr, uniq_bas_idx, inv_idx, l_ctr_counts = np.unique(
            l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)
        if mol.verbose >= logger.DEBUG:
            logger.debug1(mol, 'Number of shells for each [l, nctr] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug(mol, '    %s : %s', l_ctr, n)

        if uniq_l_ctr[:,0].max() > LMAX_ON_GPU:
            raise ValueError('High angular basis not supported')

        sorted_idx = np.argsort(inv_idx)
        ao_loc = pmol.ao_loc_nr(cart=True)
        nao = ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), ao_loc[1:-1])
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        pmol._bas = pmol._bas[sorted_idx]

        self.mol = pmol
        self.coeff = coeff[ao_idx]
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts)).astype(np.int32)

    @contextlib.contextmanager
    def gdft_envs_cache(self):
        mol = self.mol
        ao_loc = mol.ao_loc_nr(cart=True)
        libgdft.GDFTinit_envs(
            ctypes.byref(self.envs_cache), ao_loc.ctypes.data_as(ctypes.c_void_p),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol._env.size))
        try:
            yield
        finally:
            libgdft.GDFTdel_envs(ctypes.byref(self.envs_cache))

class _GDFTEnvsCache(ctypes.Structure):
    pass
