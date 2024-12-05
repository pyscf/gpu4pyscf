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
from functools import lru_cache
import contextlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cupy
from pyscf import gto, lib, dft
from pyscf.dft import numint
from pyscf.gto.eval_gto import NBINS, CUTOFF
from gpu4pyscf.gto.mole import basis_seg_contraction
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, load_library, add_sparse, release_gpu_stack, transpose_sum,
    grouped_dot, grouped_gemm, reduce_to_device)
from gpu4pyscf.dft import xc_deriv, xc_alias, libxc
from gpu4pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, _num_devices

LMAX_ON_GPU = 6
BAS_ALIGNED = 1
GRID_BLKSIZE = 32
MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 64*64)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)
AO_ALIGNMENT = getattr(__config__, 'ao_aligned', 16)
AO_THRESHOLD = 1e-10

# Should we release the cupy cache?
FREE_CUPY_CACHE = False

libgdft = load_library('libgdft')
libgdft.GDFTeval_gto.restype = ctypes.c_int
libgdft.GDFTcontract_rho.restype = ctypes.c_int
libgdft.GDFTscale_ao.restype = ctypes.c_int
libgdft.GDFTdot_ao_dm_sparse.restype = ctypes.c_int
libgdft.GDFTdot_ao_ao_sparse.restype = ctypes.c_int
libgdft.GDFTdot_aow_ao_sparse.restype = ctypes.c_int

def eval_ao(mol, coords, deriv=0, shls_slice=None, nao_slice=None, ao_loc_slice=None,
            non0tab=None, out=None, verbose=None, ctr_offsets_slice=None, gdftopt=None,
            transpose=True):
    ''' evaluate ao values for given coords and shell indices
    Kwargs:
        shls_slice :       offsets of shell slices to be evaluated
        ao_loc_slice:      offsets of ao slices to be evaluated
        ctr_offsets_slice: offsets of contraction patterns
    Returns:
        ao: comp x nao_slice x ngrids, ao is in C-contiguous.
            comp x ngrids x nao_slice if tranpose, be compatiable with PySCF.
    '''
    if gdftopt is None:
        opt = _GDFTOpt.from_mol(mol)
        with opt.gdft_envs_cache():
            return eval_ao(
                mol, coords, deriv, shls_slice, nao_slice, ao_loc_slice,
                non0tab, out, verbose, ctr_offsets_slice, opt, transpose)

    opt = gdftopt
    _sorted_mol = opt._sorted_mol

    if shls_slice is None:
        shls_slice = cupy.arange(_sorted_mol.nbas, dtype=np.int32)
        ctr_offsets = opt.l_ctr_offsets
        ctr_offsets_slice = opt.l_ctr_offsets
        ao_loc_slice = cupy.asarray(_sorted_mol.ao_loc_nr())
        nao_slice = _sorted_mol.nao
    else:
        assert ao_loc_slice is not None
        assert nao_slice is not None
        assert ctr_offsets_slice is not None
        ctr_offsets = opt.l_ctr_offsets

    nctr = ctr_offsets.size - 1
    ngrids = coords.shape[0]
    coords = cupy.asarray(coords.T, order='C')
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    stream = cupy.cuda.get_current_stream()
    
    # ao must be set to zero due to implementation
    if deriv > 1:
        if out is None:
            out = cupy.zeros((comp, nao_slice, ngrids), order='C')
        else:
            out[:] = 0
    else:
        if out is None:
            out = cupy.empty((comp, nao_slice, ngrids), order='C')

    err = libgdft.GDFTeval_gto(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(deriv), ctypes.c_int(_sorted_mol.cart),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p), ctypes.c_int(ngrids),
        ctypes.cast(shls_slice.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao_loc_slice.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao_slice),
        ctr_offsets.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nctr),
        ctr_offsets_slice.ctypes.data_as(ctypes.c_void_p),
        _sorted_mol._bas.ctypes.data_as(ctypes.c_void_p))

    if err != 0:
        raise RuntimeError('CUDA Error in evaluating AO')

    if mol is not _sorted_mol:
        coeff = cupy.asarray(opt.coeff)
        out = contract('nig,ij->njg', out, coeff)

    if transpose:
        out = out.transpose(0,2,1)
    
    if deriv == 0:
        out = out[0]
    return out

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=False, verbose=None):
    xctype = xctype.upper()
    if xctype in ('LDA', 'HF'):
        _, ngrids = ao.shape
    else:
        _, ngrids = ao[0].shape

    dm = cupy.asarray(dm)
    if xctype in ('LDA', 'HF'):
        c0 = dm.dot(ao)
        rho = _contract_rho(c0, ao)
    elif xctype in ('GGA', 'NLC'):
        rho = cupy.empty((4,ngrids))
        c0 = dm.dot(ao[0])
        rho[0] = _contract_rho(c0, ao[0])
        for i in range(1, 4):
            rho[i] = _contract_rho(c0, ao[i])
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            c0 = dm.T.dot(ao[0])
            for i in range(1, 4):
                rho[i] += _contract_rho(ao[i], c0)
    else:  # meta-GGA
        assert not with_lapl
        rho = cupy.empty((5,ngrids))
        tau_idx = 4
        c0 = dm.dot(ao[0])
        rho[0] = _contract_rho(c0, ao[0])

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = dm.dot(ao[i])
            rho[tau_idx] += _contract_rho(c1, ao[i])
            rho[i] = _contract_rho(c0, ao[i])
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho(c1, ao[0])
        rho[tau_idx] *= .5  # tau = 1/2 (\nabla f)^2

    return rho

def eval_rho1(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              with_lapl=False, verbose=None):
    raise NotImplementedError

def eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              with_lapl=False, verbose=None, out=None):
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        _, ngrids = ao.shape
    else:
        _, ngrids = ao[0].shape

    cpos = (mo_coeff * mo_occ**0.5)[:,mo_occ>0]
    if xctype == 'LDA' or xctype == 'HF':
        c0 = cupy.dot(cpos.T, ao)
        rho = _contract_rho(c0, c0)
    elif xctype in ('GGA', 'NLC'):
        rho = cupy.empty((4,ngrids))
        c0 = cupy.dot(cpos.T, ao[0])
        _contract_rho(c0, c0, rho=rho[0])
        for i in range(1, 4):
            c1 = cupy.dot(cpos.T, ao[i])
            _contract_rho(c0, c1, rho=rho[i])
        rho[1:] *= 2
    else: # meta-GGA
        assert not with_lapl
        rho = cupy.empty((5,ngrids))
        tau_idx = 4

        c0 = cupy.dot(cpos.T, ao[0])
        _contract_rho(c0, c0, rho=rho[0])
        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = cupy.dot(cpos.T, ao[i])
            rho[i] = _contract_rho(c0, c1)
            rho[tau_idx] += _contract_rho(c1, c1)

        rho[1:4] *= 2
        rho[tau_idx] *= .5
    return rho

def eval_rho3(mol, ao, c0, mo1, non0tab=None, xctype='LDA',
              with_lapl=False, verbose=None):
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        _, ngrids = ao.shape
    else:
        _, ngrids = ao[0].shape
    shls_slice = (0, mol.nbas)
    ao_loc = None #mol.ao_loc_nr()

    cpos1= mo1
    if xctype == 'LDA' or xctype == 'HF':
        c_0 = _dot_ao_dm(mol, ao, cpos1, non0tab, shls_slice, ao_loc)
        rho = _contract_rho(c0, c_0)
        rho *= 2.0
    elif xctype in ('GGA', 'NLC'):
        rho = cupy.empty((4,ngrids))
        c_0 = contract('nig,io->nog', ao, cpos1)
        _contract_rho(c0[0], c_0[0], rho=rho[0])
        for i in range(1, 4):
            _contract_rho(c_0[0], c0[i], rho=rho[i])
            rho[i] += _contract_rho(c0[0], c_0[i])
        rho *= 2.0
    else: # meta-GGA
        assert not with_lapl
        rho = cupy.empty((5,ngrids))
        tau_idx = 4
        c_0 = contract('nig,io->nog', ao, cpos1)
        #:rho[0] = numpy.einsum('pi,pi->p', c0, c0)
        rho[0] = _contract_rho(c0[0], c_0[0])
        rho[tau_idx] = 0
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
            #:rho[5] += numpy.einsum('pi,pi->p', c1, c1)
            rho[i] = _contract_rho(c_0[0], c0[i])
            rho[i]+= _contract_rho(c0[0], c_0[i])
            rho[tau_idx] += _contract_rho(c_0[i], c0[i])
        rho *= 2.0
        rho[tau_idx] *= .5
    return rho

def eval_rho4(mol, ao, mo0, mo1, non0tab=None, xctype='LDA', hermi=0,
              with_lapl=False, verbose=None):
    '''Evaluate density using first order orbitals. This density is typically
    derived from the non-symmetric density matrix (hermi=0) in TDDFT
    dm[i] = mo0.dot(mo1[i].T) and symmetric density matrix (hermi=1) in CPHF
    dm[i] = mo0.dot(mo1[i].T) + mo1[i].dot(mo0.T)

    ao: nd x nao x ng
    mo0: nao x nocc
    mo1: na x nao x nocc
    '''
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        _, ngrids = ao.shape
    else:
        _, ngrids = ao[0].shape

    na = mo1.shape[0]
    if xctype == 'LDA' or xctype == 'HF':
        c0 = mo0.T.dot(ao)
        t1 = log.timer_debug2('eval occ_coeff', *t0)
        c_0 = contract('aio,ig->aog', mo1, ao)
        rho = cupy.empty([na,ngrids])
        for i in range(na):
            rho[i] = _contract_rho(c0, c_0[i])
    elif xctype in ('GGA', 'NLC'):
        c0 = contract('nig,io->nog', ao, mo0)
        t1 = log.timer_debug2('eval occ_coeff', *t0)
        c_0 = contract('nig,aio->anog', ao, mo1)
        t1 = log.timer_debug2('ao * cpos', *t1)
        rho = cupy.empty([na, 4, ngrids])
        for i in range(na):
            _contract_rho_gga(c0, c_0[i], rho=rho[i])
    else: # meta-GGA
        assert not with_lapl
        rho = cupy.empty((na,5,ngrids))
        c0 = contract('nig,io->nog', ao, mo0)
        c_0 = contract('nig,aio->anog', ao, mo1)
        for i in range(na):
            _contract_rho_mgga(c0, c_0[i], rho=rho[i])
    if hermi:
        # corresponding to the density of ao * mo1[i].dot(mo0.T) * ao
        rho *= 2.
    t0 = log.timer_debug2('contract rho', *t0)
    return rho

def _vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    thresh=1e-10

    #output
    exc=cupy.zeros(rho[0,:].size)
    vxc=cupy.zeros([2,rho[0,:].size])

    #outer grid needs threshing
    threshind=rho[0,:]>=thresh
    coords=coords[threshind]
    R=rho[0,:][threshind]
    Gx=rho[1,:][threshind]
    Gy=rho[2,:][threshind]
    Gz=rho[3,:][threshind]
    G=Gx**2.+Gy**2.+Gz**2.

    #inner grid needs threshing
    innerthreshind=vvrho[0,:]>=thresh
    vvcoords=vvcoords[innerthreshind]
    vvweight=vvweight[innerthreshind]
    Rp=vvrho[0,:][innerthreshind]
    RpW=Rp*vvweight
    Gxp=vvrho[1,:][innerthreshind]
    Gyp=vvrho[2,:][innerthreshind]
    Gzp=vvrho[3,:][innerthreshind]
    Gp=Gxp**2.+Gyp**2.+Gzp**2.

    #constants and parameters
    Pi=cupy.pi
    Pi43=4.*Pi/3.
    Bvv, Cvv = nlc_pars
    Kvv=Bvv*1.5*Pi*((9.*Pi)**(-1./6.))
    Beta=((3./(Bvv*Bvv))**(0.75))/32.

    #inner grid
    W0p=Gp/(Rp*Rp)
    W0p=Cvv*W0p*W0p
    W0p=(W0p+Pi43*Rp)**0.5
    Kp=Kvv*(Rp**(1./6.))

    #outer grid
    W0tmp=G/(R**2)
    W0tmp=Cvv*W0tmp*W0tmp
    W0=(W0tmp+Pi43*R)**0.5
    dW0dR=(0.5*Pi43*R-2.*W0tmp)/W0
    dW0dG=W0tmp*R/(G*W0)
    K=Kvv*(R**(1./6.))
    dKdR=(1./6.)*K

    vvcoords = cupy.asarray(vvcoords, order='F')
    coords = cupy.asarray(coords, order='F')

    F = cupy.empty_like(R)
    U = cupy.empty_like(R)
    W = cupy.empty_like(R)

    #for i in range(R.size):
    #    DX=vvcoords[:,0]-coords[i,0]
    #    DY=vvcoords[:,1]-coords[i,1]
    #    DZ=vvcoords[:,2]-coords[i,2]
    #    R2=DX*DX+DY*DY+DZ*DZ
    #    gp=R2*W0p+Kp
    #    g=R2*W0[i]+K[i]
    #    gt=g+gp
    #    T=RpW/(g*gp*gt)
    #    F=numpy.sum(T)
    #    T*=(1./g+1./gt)
    #    U=numpy.sum(T)
    #    W=numpy.sum(T*R2)
    #    F*=-1.5

    stream = cupy.cuda.get_current_stream()
    err = libgdft.VXC_vv10nlc(ctypes.cast(stream.ptr, ctypes.c_void_p),
                        ctypes.cast(F.data.ptr, ctypes.c_void_p),
                        ctypes.cast(U.data.ptr, ctypes.c_void_p),
                        ctypes.cast(W.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vvcoords.data.ptr, ctypes.c_void_p),
                        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
                        ctypes.cast(W0p.data.ptr, ctypes.c_void_p),
                        ctypes.cast(W0.data.ptr, ctypes.c_void_p),
                        ctypes.cast(K.data.ptr, ctypes.c_void_p),
                        ctypes.cast(Kp.data.ptr, ctypes.c_void_p),
                        ctypes.cast(RpW.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(vvcoords.shape[0]),
                        ctypes.c_int(coords.shape[0]))

    if err != 0:
        raise RuntimeError('CUDA Error')

    #exc is multiplied by Rho later
    exc[threshind] = Beta+0.5*F
    vxc[0,threshind] = Beta+F+1.5*(U*dKdR+W*dW0dR)
    vxc[1,threshind] = 1.5*W*dW0dG
    return exc,vxc

def _nr_rks_task(ni, mol, grids, xc_code, dms, mo_coeff, mo_occ, 
                 verbose=None, with_lapl=False, grid_range=(), device_id=0):
    ''' nr_rks task on given device
    '''
    with cupy.cuda.Device(device_id), _streams[device_id]:
        if dms is not None: dms = cupy.asarray(dms)
        if mo_coeff is not None: mo_coeff = cupy.asarray(mo_coeff)
        if mo_occ is not None: mo_occ = cupy.asarray(mo_occ)

        log = logger.new_logger(mol)
        t0 = log.init_timer()
        xctype = ni._xc_type(xc_code)
        nao = mol.nao
        opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol
        nset = dms.shape[0]
        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1
        
        ngrids_glob = grids.coords.shape[0]
        ngrids_per_device = (ngrids_glob + _num_devices - 1) // _num_devices
        grid_start = device_id * ngrids_per_device
        grid_end = (device_id + 1) * ngrids_per_device
        ngrids_local = grid_end - grid_start
        
        weights = cupy.empty([ngrids_local])
        if xctype == 'LDA':
            rho_tot = cupy.empty([nset,1,ngrids_local])
        elif xctype == 'GGA':
            rho_tot = cupy.empty([nset,4,ngrids_local])
        else:
            rho_tot = cupy.empty([nset,5,ngrids_local])

        p0 = p1 = 0
        for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                     max_memory=None, 
                                                     grid_range=(grid_start, grid_end)):
            p1 = p0 + weight.size
            weights[p0:p1] = weight
            for i in range(nset):
                if mo_coeff is None:
                    rho_tot[i,:,p0:p1] = eval_rho(_sorted_mol, ao_mask, dms[i][idx[:,None],idx], 
                                                xctype=xctype, hermi=1, with_lapl=with_lapl)
                else:
                    mo_coeff_mask = mo_coeff[idx,:]
                    rho_tot[i,:,p0:p1] = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask, mo_occ, 
                                                None, xctype, with_lapl)
            p0 = p1
        t0 = log.timer_debug1(f'eval rho on Device {device_id}', *t0)
        
        # libxc calls are still running on default stream
        nelec = cupy.zeros(nset)
        excsum = cupy.zeros(nset)
        wv = []
        for i in range(nset):
            if xctype == 'LDA':
                exc, vxc = ni.eval_xc_eff(xc_code, rho_tot[i][0], deriv=1, xctype=xctype)[:2]
            else:
                exc, vxc = ni.eval_xc_eff(xc_code, rho_tot[i], deriv=1, xctype=xctype)[:2]
            vxc = cupy.asarray(vxc, order='C')
            exc = cupy.asarray(exc, order='C')
            den = rho_tot[i][0] * weights
            nelec[i] = den.sum()
            excsum[i] = cupy.dot(den, exc[:,0])
            wv.append(vxc * weights)
            if xctype == 'GGA':
                wv[i][0] *= .5
            if xctype == 'MGGA':
                wv[i][[0,4]] *= .5
        t0 = log.timer_debug1(f'eval vxc on Device {device_id}', *t0)

        vmat = cupy.zeros((nset, nao, nao))
        p0 = p1 = 0
        for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                     max_memory=None, 
                                                     grid_range=(grid_start, grid_end)):
            p1 = p0 + weight.size
            for i in range(nset):
                if xctype == 'LDA':
                    aow = _scale_ao(ao_mask, wv[i][0,p0:p1])
                    add_sparse(vmat[i], ao_mask.dot(aow.T), idx)
                elif xctype == 'GGA':
                    aow = _scale_ao(ao_mask, wv[i][:,p0:p1])
                    add_sparse(vmat[i], ao_mask[0].dot(aow.T), idx)
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                elif xctype == 'MGGA':
                    aow = _scale_ao(ao_mask, wv[i][:4,p0:p1])
                    vtmp = ao_mask[0].dot(aow.T)
                    vtmp+= _tau_dot(ao_mask, ao_mask, wv[i][4,p0:p1])
                    add_sparse(vmat[i], vtmp, idx)
                elif xctype == 'HF':
                    pass
                else:
                    raise NotImplementedError(f'numint.nr_rks for functional {xc_code}')
            p0 = p1
        t0 = log.timer_debug1(f'eval integration on {device_id}', *t0)
    return vmat, nelec.get(), excsum.get()

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    nao = mol.nao
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms,'mo_occ', None)
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    dms = opt.sort_orbitals(dms.reshape(-1,nao,nao), axis=[1,2])

    if mo_coeff is not None:
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    release_gpu_stack()
    cupy.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _nr_rks_task,
                ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                verbose=verbose, device_id=device_id)
            futures.append(future)
    vmat_dist = []
    nelec_dist = []
    excsum_dist = []
    for future in futures:
        v, n, e = future.result()
        vmat_dist.append(v)
        nelec_dist.append(n)
        excsum_dist.append(e)
    vmat = reduce_to_device(vmat_dist, inplace=True)
    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    nelec = np.sum(nelec_dist, axis=0)
    excsum = np.sum(excsum_dist, axis=0)

    if xctype != 'LDA':
        transpose_sum(vmat)

    if FREE_CUPY_CACHE:
        dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    t0 = log.timer_debug1('nr_rks', *t0)
    return nelec, excsum, vmat

def eval_rho_group(mol, ao_group, mo_coeff_group, mo_occ, 
                   non0tab=None, xctype='LDA',
                   with_lapl=False, verbose=None, out=None):
    groups = len(ao_group)
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids_group = []
        for ao in ao_group:
            _, ngrids = ao.shape
            ngrids_group.append(ngrids)
    else:
        ngrids_group = []
        for ao in ao_group:
            _, ngrids = ao[0].shape
            ngrids_group.append(ngrids)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    cpos_group = []
    for groups_idx in range(groups):
        cpos = (mo_coeff_group[groups_idx] * mo_occ**0.5)[:,mo_occ>0]
        cpos_group.append(cpos)
    if xctype == 'LDA' or xctype == 'HF':
        c0_group = grouped_gemm(cpos_group, ao_group)
        rho_group = []
        for c0 in c0_group:
            rho = _contract_rho(c0, c0)
            rho_group.append(rho)
    elif xctype in ('GGA', 'NLC'):
        c0_group = []
        cpos_group4 = []
        ao_group4 = []
        for ao, cpos in zip(ao_group, cpos_group):
            for i in range(4):
                cpos_group4.append(cpos)
                ao_group4.append(ao[i])
        c0_group = grouped_gemm(cpos_group4, ao_group4)

        rho_group = []
        for groups_idx in range(groups):
            rho = cupy.empty((4, ngrids_group[groups_idx]))
            c0 = c0_group[4*groups_idx:4*(groups_idx+1)]
            _contract_rho(c0[0], c0[0], rho=rho[0])
            for i in range(1, 4):
                _contract_rho(c0[0], c0[i], rho=rho[i])
            rho[1:] *= 2
            rho_group.append(rho)
    else: # meta-GGA
        assert not with_lapl
        c0_group = []
        cpos_group4 = []
        ao_group4 = []
        for ao, cpos in zip(ao_group, cpos_group):
            for i in range(4):
                cpos_group4.append(cpos)
                ao_group4.append(ao[i])
        c0_group = grouped_gemm(cpos_group4, ao_group4)

        rho_group = []
        for groups_idx in range(groups):
            ngrids = ngrids_group[groups_idx]
            c0 = c0_group[4*groups_idx:4*(groups_idx+1)]
            if with_lapl:
                rho = cupy.empty((6, ngrids))
                tau_idx = 5
            else:
                rho = cupy.empty((5, ngrids))
                tau_idx = 4
            _contract_rho(c0[0], c0[0], rho=rho[0])
            rho[tau_idx] = 0
            for i in range(1, 4):
                _contract_rho(c0[0], c0[i], rho[i])
                rho[tau_idx] += _contract_rho(c0[i], c0[i])

            if with_lapl:
                ao = ao_group[groups_idx]
                if ao.shape[0] > 4:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(mol, ao2, cpos, non0tab, shls_slice, ao_loc)
                    rho[4] = _contract_rho(c0[0], c1)
                    rho[4] += rho[5]
                    rho[4] *= 2
                else:
                    rho[4] = 0
            rho[1:4] *= 2
            rho[tau_idx] *= .5
            rho_group.append(rho)
    return rho_group

def nr_rks_group(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms,'mo_occ', None)

    mol = None
    _sorted_mol = opt._sorted_mol
    nao, nao0 = opt.coeff.shape
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    dms = opt.sort_orbitals(dms.reshape(-1,nao0,nao0), axis=[1,2])
    nset = len(dms)

    if mo_coeff is not None:
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    nelec = cupy.zeros(nset)
    excsum = cupy.zeros(nset)
    vmat = cupy.zeros((nset, nao, nao))

    release_gpu_stack()
    if xctype == 'LDA':
        ao_deriv = 0
    else:
        ao_deriv = 1
    ngrids = grids.weights.size
    if xctype == 'LDA':
        rho_tot = cupy.empty([nset,1,ngrids])
    elif xctype == 'GGA':
        rho_tot = cupy.empty([nset,4,ngrids])
    else:
        rho_tot = cupy.empty([nset,5,ngrids])
    p0 = p1 = 0
    t1 = t0 = log.init_timer()
    for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, 
                                                 max_memory=max_memory):
        p1 = p0 + weight.size
        for i in range(nset):
            if mo_coeff is None:
                rho_tot[i,:,p0:p1] = eval_rho(
                    _sorted_mol, ao_mask, dms[i][idx[:,None],idx], 
                    xctype=xctype, hermi=1)
            else:
                mo_coeff_mask = mo_coeff[idx,:]
                rho_tot[i,:,p0:p1] = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask, 
                                               mo_occ, None, xctype)
        p0 = p1
        t1 = log.timer_debug2('eval rho slice', *t1)
    t0 = log.timer_debug1('eval rho', *t0)

    wv = []
    for i in range(nset):
        if xctype == 'LDA':
            exc, vxc = ni.eval_xc_eff(xc_code, rho_tot[i][0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = ni.eval_xc_eff(xc_code, rho_tot[i], deriv=1, xctype=xctype)[:2]
        vxc = cupy.asarray(vxc, order='C')
        exc = cupy.asarray(exc, order='C')
        den = rho_tot[i][0] * grids.weights
        nelec[i] = den.sum()
        excsum[i] = cupy.sum(den * exc[:,0])
        wv.append(vxc * grids.weights)
        if xctype == 'GGA':
            wv[i][0] *= .5
        if xctype == 'MGGA':
            wv[i][[0,4]] *= .5
    t0 = log.timer_debug1('eval vxc', *t0)

    t1 = t0
    p0 = p1 = 0
    for ao_mask_group, idx_group, weight_group, _ in ni.grouped_block_loop(_sorted_mol, grids, nao, ao_deriv):
        p0_raw = p0
        for i in range(nset):
            p0 = p0_raw
            if xctype == 'LDA':
                aow_group = []
                for weight, ao_mask in zip(weight_group, ao_mask_group):
                    p1 = p0 + weight.size
                    aow = _scale_ao(ao_mask, wv[i][0,p0:p1])
                    p0 = p1
                    aow_group.append(aow)
                dot_res_group = grouped_dot(ao_mask_group, aow_group)
                for dot_res, idx in zip(dot_res_group, idx_group):
                    add_sparse(vmat[i], dot_res, idx)
            elif xctype == 'GGA':
                aow_group = []
                ao0_mask_group = []
                for weight, ao_mask in zip(weight_group, ao_mask_group):
                    p1 = p0 + weight.size
                    aow = _scale_ao(ao_mask, wv[i][:,p0:p1])
                    p0 = p1
                    aow_group.append(aow)
                    ao0_mask_group.append(ao_mask[0])
                vmat_group = grouped_dot(ao0_mask_group, aow_group)
                for vmat_tmp, idx in zip(vmat_group, idx_group):
                    add_sparse(vmat[i], vmat_tmp, idx)
            elif xctype == 'NLC':
                raise NotImplementedError('NLC')
            elif xctype == 'MGGA':
                aow_group = []
                ao0_mask_group = []
                p0_tmp = p0
                for weight, ao_mask in zip(weight_group, ao_mask_group):
                    p1 = p0 + weight.size
                    aow = _scale_ao(ao_mask, wv[i][:4,p0:p1])
                    p0 = p1
                    aow_group.append(aow)
                    ao0_mask_group.append(ao_mask[0])
                vmat_group = grouped_dot(ao0_mask_group, aow_group)
                p0 = p0_tmp
                for weight, vmat_tmp, ao_mask, idx in zip(weight_group, vmat_group, ao_mask_group, idx_group):
                    p1 = p0 + weight.size
                    vmat_tmp += _tau_dot(ao_mask, ao_mask, wv[i][4,p0:p1])
                    add_sparse(vmat[i], vmat_tmp, idx)
                    p0 = p1
            elif xctype == 'HF':
                pass
            else:
                raise NotImplementedError(f'numint.nr_rks for functional {xc_code}')
        t1 = log.timer_debug2('integration', *t1)
    t0 = log.timer_debug1('vxc integration', *t0)
    vmat = opt.unsort_orbitals(vmat, axis=[1,2])

    if xctype != 'LDA':
        transpose_sum(vmat)

    if FREE_CUPY_CACHE:
        dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    return nelec, excsum, vmat

def _nr_uks_task(ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                verbose=None, with_lapl=False, grid_range=(), device_id=0):
    ''' nr_uks task on one device
    '''
    with cupy.cuda.Device(device_id), _streams[device_id]:
        if dms is not None:
            dma, dmb = dms 
            dma = cupy.asarray(dma)
            dmb = cupy.asarray(dmb)
        if mo_coeff is not None: mo_coeff = cupy.asarray(mo_coeff)
        if mo_occ is not None: mo_occ = cupy.asarray(mo_occ)
        
        log = logger.new_logger(mol)
        t0 = log.init_timer()
        xctype = ni._xc_type(xc_code)
        nao = mol.nao
        opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol
        
        nset = dma.shape[0]
        nelec = np.zeros((2,nset))
        excsum = np.zeros(nset)
        vmata = cupy.zeros((nset, nao, nao))
        vmatb = cupy.zeros((nset, nao, nao))

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids_glob = grids.coords.shape[0]
        ngrids_per_device = (ngrids_glob + _num_devices - 1) // _num_devices
        grid_start = device_id * ngrids_per_device
        grid_end = (device_id + 1) * ngrids_per_device

        for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                     max_memory=None, 
                                                     grid_range=(grid_start, grid_end)):
            for i in range(nset):
                t0 = log.init_timer()
                if mo_coeff is None:
                    rho_a = eval_rho(_sorted_mol, ao_mask, dma[i][idx[:,None],idx], xctype=xctype, hermi=1)
                    rho_b = eval_rho(_sorted_mol, ao_mask, dmb[i][idx[:,None],idx], xctype=xctype, hermi=1)
                else:
                    mo_coeff_mask = mo_coeff[:, idx,:]
                    rho_a = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask[0], mo_occ[0], None, xctype)
                    rho_b = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask[1], mo_occ[1], None, xctype)

                rho = cupy.stack([rho_a, rho_b], axis=0)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
                t1 = log.timer_debug1('eval vxc', *t0)
                if xctype == 'LDA':
                    den_a = rho_a * weight
                    den_b = rho_b * weight
                    wv = vxc[:,0] * weight
                    va = ao_mask.dot(_scale_ao(ao_mask, wv[0]).T)
                    vb = ao_mask.dot(_scale_ao(ao_mask, wv[1]).T)
                    add_sparse(vmata[i], va, idx)
                    add_sparse(vmatb[i], vb, idx)

                elif xctype == 'GGA':
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                    wv = vxc * weight
                    wv[:,0] *= .5
                    va = ao_mask[0].dot(_scale_ao(ao_mask, wv[0]).T)
                    vb = ao_mask[0].dot(_scale_ao(ao_mask, wv[1]).T)
                    add_sparse(vmata[i], va, idx)
                    add_sparse(vmatb[i], vb, idx)
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                elif xctype == 'MGGA':
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                    wv = vxc * weight
                    wv[:,[0, 4]] *= .5
                    va = ao_mask[0].dot(_scale_ao(ao_mask[:4], wv[0,:4]).T)
                    vb = ao_mask[0].dot(_scale_ao(ao_mask[:4], wv[1,:4]).T)
                    va += _tau_dot(ao_mask, ao_mask, wv[0,4])
                    vb += _tau_dot(ao_mask, ao_mask, wv[1,4])
                    add_sparse(vmata[i], va, idx)
                    add_sparse(vmatb[i], vb, idx)
                elif xctype == 'HF':
                    pass
                else:
                    raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')
                nelec[0,i] += den_a.sum()
                nelec[1,i] += den_b.sum()
                excsum[i] += cupy.dot(den_a, exc[:,0])
                excsum[i] += cupy.dot(den_b, exc[:,0])
                t1 = log.timer_debug1('integration', *t1)
    
    return nelec, excsum, (vmata, vmatb)

def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms,'mo_occ', None)
    nao = mol.nao
    dma, dmb = dms
    dm_shape = dma.shape
    dma = cupy.asarray(dma).reshape(-1,nao,nao)
    dmb = cupy.asarray(dmb).reshape(-1,nao,nao)
    dma = opt.sort_orbitals(dma, axis=[1,2])
    dmb = opt.sort_orbitals(dmb, axis=[1,2])
    nset = len(dma)

    if mo_coeff is not None:
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    release_gpu_stack()
    cupy.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _nr_uks_task,
                ni, mol, grids, xc_code, (dma,dmb), mo_coeff, mo_occ,
                verbose=verbose, device_id=device_id)
            futures.append(future)

    vmata_dist = []
    vmatb_dist = []
    nelec_dist = []
    excsum_dist = []
    for future in futures:
        n, e, v = future.result()
        vmata_dist.append(v[0])
        vmatb_dist.append(v[1])
        nelec_dist.append(n)
        excsum_dist.append(e)
        
    vmata = reduce_to_device(vmata_dist, inplace=True)
    vmatb = reduce_to_device(vmatb_dist, inplace=True)
    vmata = opt.unsort_orbitals(vmata, axis=[1,2])
    vmatb = opt.unsort_orbitals(vmatb, axis=[1,2])
    
    nelec = np.sum(nelec_dist, axis=0)
    excsum = np.sum(excsum_dist, axis=0)

    if xctype != 'LDA':
        for i in range(nset):
            vmata[i] = vmata[i] + vmata[i].T
            vmatb[i] = vmatb[i] + vmatb[i].T

    if FREE_CUPY_CACHE:
        dma = dmb = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        nelec = nelec.reshape(2)
        excsum = excsum[0]
        vmata = vmata[0]
        vmatb = vmatb[0]
    vmat = cupy.asarray([vmata, vmatb])
    return nelec, excsum, vmat


def get_rho(ni, mol, dm, grids, max_memory=2000, verbose=None):
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mol = None
    _sorted_mol = opt._sorted_mol
    log = logger.new_logger(opt.mol, verbose)
    coeff = cupy.asarray(opt.coeff)
    nao = coeff.shape[0]
    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm,'mo_occ', None)

    dm = coeff @ cupy.asarray(dm) @ coeff.T
    if mo_coeff is not None:
        mo_coeff = coeff @ mo_coeff

    mem_avail = get_avail_mem()
    blksize = mem_avail*.2/8/nao//ALIGNED * ALIGNED
    blksize = min(blksize, MIN_BLK_SIZE)
    GB = 1024*1024*1024
    log.debug(f'GPU Memory {mem_avail/GB:.1f} GB available, block size {blksize}')

    ngrids = grids.weights.size
    rho = cupy.empty(ngrids)
    with opt.gdft_envs_cache():
        t1 = t0 = log.init_timer()
        for p0, p1 in lib.prange(0,ngrids,blksize):
            coords = grids.coords[p0:p1]
            ao = eval_ao(_sorted_mol, coords, 0, gdftopt=opt, transpose=False)
            if mo_coeff is None:
                rho[p0:p1] = eval_rho(_sorted_mol, ao, dm, xctype='LDA', hermi=1)
            else:
                rho[p0:p1] = eval_rho2(_sorted_mol, ao, mo_coeff, mo_occ, None, 'LDA')
            t1 = log.timer_debug2('eval rho slice', *t1)
    t0 = log.timer_debug1('eval rho', *t0)

    if FREE_CUPY_CACHE:
        dm = None
        cupy.get_default_memory_pool().free_all_blocks()
    return rho

def _nr_rks_fxc_task(ni, mol, grids, xc_code, fxc, dms, mo1, occ_coeff,
                     verbose=None, hermi=1, device_id=0):
    with cupy.cuda.Device(device_id), _streams[device_id]:
        if dms is not None: dms = cupy.asarray(dms)
        if mo1 is not None: mo1 = cupy.asarray(mo1)
        if occ_coeff is not None: occ_coeff = cupy.asarray(occ_coeff)
        if fxc is not None: fxc = cupy.asarray(fxc)

        log = logger.new_logger(mol, verbose)
        xctype = ni._xc_type(xc_code)
        opt = getattr(ni, 'gdftopt', None)

        _sorted_mol = opt.mol
        nao = mol.nao
        dms = cupy.asarray(dms)
        nset = len(dms)
        vmat = cupy.zeros((nset, nao, nao))

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids_glob = grids.coords.shape[0]
        ngrids_per_device = (ngrids_glob + _num_devices - 1) // _num_devices
        grid_start = device_id * ngrids_per_device
        grid_end = (device_id + 1) * ngrids_per_device

        p0 = p1 = grid_start
        t1 = t0 = log.init_timer()
        for ao, mask, weights, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                       max_memory=None,
                                                       grid_range=(grid_start, grid_end)):
            p0, p1 = p1, p1+len(weights)
            # precompute molecular orbitals
            if occ_coeff is not None:
                occ_coeff_mask = occ_coeff[mask]
                rho1 = eval_rho4(_sorted_mol, ao, occ_coeff_mask, mo1[:,mask],
                                xctype=xctype, hermi=hermi)
            else:
                # slow version
                rho1 = []
                for i in range(nset):
                    rho_tmp = eval_rho(_sorted_mol, ao, dms[i,mask[:,None],mask],
                                    xctype=xctype, hermi=hermi)
                    rho1.append(rho_tmp)
                rho1 = cupy.stack(rho1, axis=0)
            t1 = log.timer_debug2('eval rho', *t1)

            # precompute fxc_w
            if xctype == 'LDA':
                fxc_w = fxc[0,0,p0:p1] * weights
                wv = rho1 * fxc_w
            else:
                fxc_w = fxc[:,:,p0:p1] * weights
                wv = contract('axg,xyg->ayg', rho1, fxc_w)

            for i in range(nset):
                if xctype == 'LDA':
                    vmat_tmp = ao.dot(_scale_ao(ao, wv[i]).T)
                elif xctype == 'GGA':
                    wv[i,0] *= .5
                    aow = _scale_ao(ao, wv[i])
                    vmat_tmp = aow.dot(ao[0].T)
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                else:
                    wv[i,0] *= .5
                    wv[i,4] *= .5
                    vmat_tmp = ao[0].dot(_scale_ao(ao[:4], wv[i,:4]).T)
                    vmat_tmp+= _tau_dot(ao, ao, wv[i,4])
                add_sparse(vmat[i], vmat_tmp, mask)

            t1 = log.timer_debug2('integration', *t1)
            ao = rho1 = None
        t0 = log.timer_debug1('vxc', *t0)
    return vmat

def nr_rks_fxc(ni, mol, grids, xc_code, dm0=None, dms=None, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    nao = mol.nao
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    # AO basis -> gdftopt AO basis
    with_mocc = hasattr(dms, 'mo1')
    mo1 = occ_coeff = None
    if with_mocc:
        mo1 = opt.sort_orbitals(dms.mo1, axis=[1])
        occ_coeff = opt.sort_orbitals(dms.occ_coeff, axis=[0]) * 2.0
    dms = opt.sort_orbitals(dms.reshape(-1,nao,nao), axis=[1,2])

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _nr_rks_fxc_task,
                ni, mol, grids, xc_code, fxc, dms, mo1, occ_coeff,
                verbose=verbose, hermi=hermi, device_id=device_id)
            futures.append(future)
    vmat_dist = []
    for future in futures:
        vmat_dist.append(future.result())
    vmat = reduce_to_device(vmat_dist, inplace=True)
    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    if xctype != 'LDA':
        transpose_sum(vmat)

    if FREE_CUPY_CACHE:
        dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        vmat = vmat[0]

    return cupy.asarray(vmat)

def nr_rks_fxc_st(ni, mol, grids, xc_code, dm0=None, dms_alpha=None,
                  relativity=0, singlet=True, rho0=None, vxc=None, fxc=None,
                  max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    if singlet:
        fxc = fxc[0,:,0] + fxc[0,:,1]
    else:
        fxc = fxc[0,:,0] - fxc[0,:,1]
    return nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms_alpha, hermi=0, fxc=fxc,
                      max_memory=max_memory, verbose=verbose)


def nr_uks_fxc(ni, mol, grids, xc_code, dm0=None, dms=None, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mol = None
    _sorted_mol = opt._sorted_mol
    nao, nao0 = opt.coeff.shape
    dma, dmb = dms
    dm_shape = dma.shape
    # AO basis -> gdftopt AO basis
    with_mocc = hasattr(dms, 'mo1')
    if with_mocc:
        mo1a, mo1b = dms.mo1
        occ_coeffa, occ_coeffb = dms.occ_coeff
        mo1a = opt.sort_orbitals(mo1a, axis=[1])
        mo1b = opt.sort_orbitals(mo1b, axis=[1])
        occ_coeff_a = opt.sort_orbitals(occ_coeffa, axis=[0])
        occ_coeff_b = opt.sort_orbitals(occ_coeffb, axis=[0])

    dma = cupy.asarray(dma).reshape(-1,nao0,nao0)
    dmb = cupy.asarray(dmb).reshape(-1,nao0,nao0)
    dma = opt.sort_orbitals(dma, axis=[1,2])
    dmb = opt.sort_orbitals(dmb, axis=[1,2])

    nset = len(dma)
    vmata = cupy.zeros((nset, nao, nao))
    vmatb = cupy.zeros((nset, nao, nao))

    if xctype == 'LDA':
        ao_deriv = 0
        nvar = 1
    elif xctype == 'GGA':
        ao_deriv = 1
        nvar = 4
    else:
        ao_deriv = 1
        nvar = 5
    p0 = p1 = 0
    for ao, mask, weights, coords in ni.block_loop(
            _sorted_mol, grids, nao, ao_deriv, max_memory=max_memory):
        t0 = log.init_timer()
        p0, p1 = p1, p1+len(weights)
        # precompute fxc_w
        fxc_w = fxc[:,:,:,:,p0:p1] * weights

        # precompute molecular orbitals
        if with_mocc:
            occ_coeff_a_mask = occ_coeff_a[mask]
            occ_coeff_b_mask = occ_coeff_b[mask]
            rho1a = eval_rho4(_sorted_mol, ao, occ_coeff_a_mask, mo1a[:,mask],
                              xctype=xctype, hermi=hermi)
            rho1b = eval_rho4(_sorted_mol, ao, occ_coeff_b_mask, mo1b[:,mask],
                              xctype=xctype, hermi=hermi)
            rho1 = cupy.stack([rho1a, rho1b]).reshape(2, nset, nvar, p1-p0)
        else: # slow version
            rho1 = cupy.empty((2, nset, nvar, p1-p0))
            for i in range(nset):
                rho1[0,i] = eval_rho(_sorted_mol, ao, dma[i,mask[:,None],mask],
                                     xctype=xctype, hermi=hermi)
                rho1[1,i] = eval_rho(_sorted_mol, ao, dmb[i,mask[:,None],mask],
                                     xctype=xctype, hermi=hermi)
        t0 = log.timer_debug1('rho', *t0)

        for i in range(nset):
            wv = contract('axg,axbyg->byg', rho1[:,i], fxc_w)
            if xctype == 'LDA':
                va = ao.dot(_scale_ao(ao, wv[0,0]).T)
                vb = ao.dot(_scale_ao(ao, wv[1,0]).T)
            elif xctype == 'GGA':
                wv[:,0] *= .5 # for transpose_sum at the end
                va = ao[0].dot(_scale_ao(ao, wv[0]).T)
                vb = ao[0].dot(_scale_ao(ao, wv[1]).T)
            elif xctype == 'NLC':
                raise NotImplementedError('NLC')
            else:
                wv[:,[0,4]] *= .5 # for transpose_sum at the end
                va = ao[0].dot(_scale_ao(ao[:4], wv[0,:4]).T)
                vb = ao[0].dot(_scale_ao(ao[:4], wv[1,:4]).T)
                va += _tau_dot(ao, ao, wv[0,4])
                vb += _tau_dot(ao, ao, wv[1,4])
            add_sparse(vmata[i], va, mask)
            add_sparse(vmatb[i], vb, mask)
    vmata = opt.unsort_orbitals(vmata, axis=[1,2])
    vmatb = opt.unsort_orbitals(vmatb, axis=[1,2])
    if xctype != 'LDA':
        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        transpose_sum(vmata)
        transpose_sum(vmatb)

    if FREE_CUPY_CACHE:
        dma = dmb = None
        cupy.get_default_memory_pool().free_all_blocks()

    if len(dm_shape) == 2:
        vmata = vmata[0]
        vmatb = vmatb[0]
    vmat = cupy.asarray([vmata, vmatb])
    return vmat

def nr_nlc_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
    '''Calculate NLC functional and potential matrix on given grids

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : 2D array
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms,'mo_occ', None)

    nao, nao0 = opt.coeff.shape
    mol = None
    _sorted_mol = opt._sorted_mol
    
    dms = dms.reshape(-1,nao0,nao0)
    assert len(dms) == 1
    dms = opt.sort_orbitals(dms, axis=[1,2])

    if mo_coeff is not None:
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    ao_deriv = 1
    vvrho = []
    for ao, idx, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=max_memory):
        if mo_coeff is None:
            rho = eval_rho(_sorted_mol, ao, dms[0][idx[:,None],idx], xctype='GGA', hermi=1)
        else:
            mo_coeff_mask = mo_coeff[idx,:]
            rho = eval_rho2(_sorted_mol, ao, mo_coeff_mask, mo_occ, None, 'GGA')
        vvrho.append(rho)

    rho = cupy.hstack(vvrho)
    t1 = log.timer_debug1('eval rho', *t0)
    exc = 0
    vxc = 0
    nlc_coefs = ni.nlc_coeff(xc_code)
    for nlc_pars, fac in nlc_coefs:
        e, v = _vv10nlc(rho, grids.coords, rho, grids.weights,
                        grids.coords, nlc_pars)
        exc += e * fac
        vxc += v * fac
    t1 = log.timer_debug1('eval vv on grids', *t1)

    den = rho[0] * grids.weights
    nelec = den.sum()
    excsum = cupy.dot(den, exc)
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)
    t1 = log.timer_debug1('transform vxc', *t1)

    vmat = cupy.zeros((nao,nao))
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5
        aow = _scale_ao(ao, wv)
        add_sparse(vmat, ao[0].dot(aow.T), mask)
    t1 = log.timer_debug1('integration', *t1)

    transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[0,1])
    log.timer_debug1('eval vv10', *t0)
    return nelec, excsum, vmat

def cache_xc_kernel(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    max_memory=2000):
    log = logger.new_logger(mol, mol.verbose)
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
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mol = None
    _sorted_mol = opt._sorted_mol
    mo_coeff = cupy.asarray(mo_coeff)
    nao = opt.coeff.shape[0]
    if mo_coeff.ndim == 2: # RHF
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
        rho = []
        t1 = t0 = log.init_timer()
        for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                     max_memory=max_memory):
            mo_coeff_mask = mo_coeff[idx,:]
            rho_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask, mo_occ, None, xctype)
            rho.append(rho_slice)
            t1 = log.timer_debug2('eval rho slice', *t1)
        rho = cupy.hstack(rho)
        if spin == 1: # RKS with nr_rks_fxc_st
            rho *= .5
            rho = cupy.repeat(rho[None], 2, axis=0)
        t0 = log.timer_debug1('eval rho in fxc', *t0)
    else:
        assert spin == 1
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])
        rhoa = []
        rhob = []
        t1 = t0 = log.init_timer()
        for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                     max_memory=max_memory):
            mo_coeff_mask = mo_coeff[:,idx,:]
            rhoa_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask[0], mo_occ[0], None, xctype)
            rhob_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff_mask[1], mo_occ[1], None, xctype)
            rhoa.append(rhoa_slice)
            rhob.append(rhob_slice)
            t1 = log.timer_debug2('eval rho in fxc', *t1)
        #rho = (cupy.hstack(rhoa), cupy.hstack(rhob))
        rho = cupy.stack([cupy.hstack(rhoa), cupy.hstack(rhob)], axis=0)
        t0 = log.timer_debug1('eval rho in fxc', *t0)    
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
    t0 = log.timer_debug1('eval fxc', *t0)
    return rho, vxc, fxc

@cupy.fuse()
def batch_square(a):
    return a[0]**2 + a[1]**2 + a[2]**2

def eval_xc_eff(ni, xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None):
    '''
    Different from PySCF, this function employ cuda version libxc
    '''
    if xctype == 'LDA':
        spin_polarized = rho.ndim >= 2
    else:
        spin_polarized = rho.ndim == 3

    if omega is None: omega = ni.omega
    if xctype is None: xctype = ni._xc_type(xc_code)

    xcfuns = ni._init_xcfuns(xc_code, spin_polarized)

    inp = {}
    if not spin_polarized:
        if xctype == 'LDA':
            inp['rho'] = rho
        if xctype == 'GGA':
            inp['rho'] = rho[0]
            inp['sigma'] = batch_square(rho[1:4])
        if xctype == 'MGGA':
            inp['rho'] = rho[0]
            inp['sigma'] = batch_square(rho[1:4])
            inp['tau'] = rho[-1]     # can be 4 (without laplacian) or 5 (with laplacian)
    else:
        if xctype == 'LDA':
            inp['rho'] = cupy.stack([rho[0], rho[1]], axis=1)
        if xctype == 'GGA':
            inp['rho'] = cupy.stack([rho[0,0], rho[1,0]], axis=1)
            sigma0 = batch_square(rho[0,1:4])
            sigma1 = rho[0,1]*rho[1,1] + rho[0,2]*rho[1,2] + rho[0,3]*rho[1,3]
            sigma2 = batch_square(rho[1,1:4])
            inp['sigma'] = cupy.stack([sigma0, sigma1, sigma2], axis=1)
        if xctype == 'MGGA':
            inp['rho'] = cupy.stack([rho[0,0], rho[1,0]], axis=1)
            sigma0 = batch_square(rho[0,1:4])
            sigma1 = rho[0,1]*rho[1,1] + rho[0,2]*rho[1,2] + rho[0,3]*rho[1,3]
            sigma2 = batch_square(rho[1,1:4])
            inp['sigma'] = cupy.stack([sigma0, sigma1, sigma2], axis=1)
            inp['tau'] = cupy.stack([rho[0,-1], rho[1,-1]], axis=1)     # can be 4 (without laplacian) or 5 (with laplacian)
    do_vxc = True
    do_fxc = deriv > 1
    do_kxc = deriv > 2

    vxc_labels = ["vrho", "vsigma", "vlapl", "vtau"]
    fxc_labels = ["v2rho2", "v2rhosigma", "v2sigma2", "v2lapl2", "v2tau2",
            "v2rholapl", "v2rhotau", "v2lapltau", "v2sigmalapl", "v2sigmatau"]
    kxc_labels = ["v3rho3", "v3rho2sigma", "v3rhosigma2", "v3sigma3",
           "v3rho2lapl", "v3rho2tau",
           "v3rhosigmalapl", "v3rhosigmatau",
           "v3rholapl2", "v3rholapltau","v3rhotau2",
           "v3sigma2lapl", "v3sigma2tau",
           "v3sigmalapl2", "v3sigmalapltau", "v3sigmatau2",
           "v3lapl3", "v3lapl2tau", "v3lapltau2", "v3tau3"]
    if len(xcfuns) == 1:
        xcfun, _ = xcfuns[0]
        xc_res = xcfun.compute(inp, do_exc=True, do_vxc=do_vxc, do_fxc=do_fxc, do_kxc=do_kxc)
        ret_full = xc_res
    else:
        ret_full = {}
        for xcfun, w in xcfuns:
            xc_res = xcfun.compute(inp, do_exc=True, do_vxc=do_vxc, do_fxc=do_fxc, do_kxc=do_kxc)
            for label in xc_res:
                if label in ret_full:
                    ret_full[label] += xc_res[label] * w
                else:
                    ret_full[label] = xc_res[label] * w
    vxc = None
    fxc = None
    kxc = None

    exc = ret_full["zk"]
    if not spin_polarized:
        vxc = [ret_full[label] for label in vxc_labels if label in ret_full]
        if do_fxc:
            fxc = [ret_full[label] for label in fxc_labels if label in ret_full]
        if do_kxc:
            kxc = [ret_full[label] for label in kxc_labels if label in ret_full]
    else:
        vxc = [ret_full[label] for label in vxc_labels if label in ret_full]
        if do_fxc:
            fxc = [ret_full[label] for label in fxc_labels if label in ret_full]
        if do_kxc:
            kxc = [ret_full[label] for label in kxc_labels if label in ret_full]
    if do_kxc:
        kxc = xc_deriv.transform_kxc(rho, fxc, kxc, xctype, spin_polarized)
    if do_fxc:
        fxc = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin_polarized)
    vxc = xc_deriv.transform_vxc(rho, vxc, xctype, spin_polarized)
    return exc, vxc, fxc, kxc

def _init_xcfuns(xc_code, spin):
    xc_upper = xc_code.upper()
    xc_names = dft.libxc.parse_xc(xc_upper)[1:][0]
    if spin:
        spin_polarized = 'polarized'
    else:
        spin_polarized = 'unpolarized'
    xcfuns = []
    for xc, w in xc_names:
        xcfun = libxc.XCfun(xc, spin_polarized)
        xcfuns.append((xcfun,w))
        if dft.libxc.needs_laplacian(xcfun.func_id):
            raise NotImplementedError()
    return xcfuns

def _sparse_index(mol, coords, l_ctr_offsets):
    '''
    determine sparse AO indices
    '''
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    stream = cupy.cuda.get_current_stream()
    cutoff = AO_THRESHOLD
    ng = coords.shape[0]
    nctr = len(l_ctr_offsets) - 1
    ao_loc = mol.ao_loc_nr()
    non0shl_idx = cupy.zeros(len(ao_loc)-1, dtype=np.int32)
    coords = cupy.asarray(coords)
    
    libgdft.GDFTscreen_index(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(non0shl_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_double(cutoff),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ng),
        l_ctr_offsets.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nctr),
        mol._bas.ctypes.data_as(ctypes.c_void_p))
    non0shl_idx = non0shl_idx.get()

    # offset of contraction pattern, used in eval_ao
    cumsum = np.cumsum(non0shl_idx, dtype=np.int32)
    glob_ctr_offsets = l_ctr_offsets
    ctr_offsets_slice = cumsum[glob_ctr_offsets-1]
    ctr_offsets_slice[0] = 0

    from pyscf import gto
    gto_type = 'cart' if mol.cart else 'sph'
    non0shl_idx = non0shl_idx == 1
    ao_loc_slice = gto.moleintor.make_loc(mol._bas[non0shl_idx,:], gto_type)
    ao_loc_slice = cupy.asarray(ao_loc_slice, dtype=np.int32)
    non0ao_idx = []
    zero_idx = []
    for sh_idx in range(len(ao_loc)-1):
        p0, p1 = ao_loc[sh_idx], ao_loc[sh_idx+1]
        if non0shl_idx[sh_idx]:
            non0ao_idx += range(p0,p1)
        else:
            zero_idx += range(p0,p1)

    idx = np.asarray(non0ao_idx, dtype=np.int32)
    zero_idx = np.asarray(zero_idx, dtype=np.int32)
    pad = (len(idx) + AO_ALIGNMENT - 1) // AO_ALIGNMENT * AO_ALIGNMENT - len(idx)
    idx = np.hstack([idx, zero_idx[:pad]])
    pad = min(pad, len(zero_idx))
    non0shl_idx = cupy.asarray(np.where(non0shl_idx)[0], dtype=np.int32)
    t1 = log.timer_debug2('init ao sparsity', *t1)
    return pad, cupy.asarray(idx), non0shl_idx, ctr_offsets_slice, ao_loc_slice

def _block_loop(ni, mol, grids, nao=None, deriv=0, max_memory=2000,
                non0tab=None, blksize=None, buf=None, extra=0, grid_range=None):
    '''
    Define this macro to loop over grids by blocks.
    Sparsity is not implemented yet
    sorted_ao: by default ao_value is sorted for GPU
    '''
    log = logger.new_logger(ni, ni.verbose)
    if grids.coords is None:
        grids.build(with_non0tab=False, sort_grids=True)
    if nao is None:
        nao = mol.nao
    
    if grid_range is None:
        grid_start, grid_end = 0, grids.coords.shape[0]
    else:
        grid_start, grid_end = grid_range
    ngrids = grid_end - grid_start

    device_id = cupy.cuda.Device().id
    log.debug(f'{grid_start} - {grid_end} grids are calculated on Device {device_id}.')
    
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if blksize is None:
        #cupy.get_default_memory_pool().free_all_blocks()
        mem_avail = get_avail_mem()
        blksize = int((mem_avail*.2/8/((comp+1)*nao + extra))/ ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        log.debug(f'{mem_avail/1e6} MB memory is available on Device {device_id}, block_size {blksize}')
        if blksize < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    coords_device = cupy.asarray(grids.coords)
    weights_device = cupy.asarray(grids.weights)
    _sorted_mol = opt._sorted_mol

    mol = None
    with opt.gdft_envs_cache():
        for block_id, (ip0, ip1) in enumerate(lib.prange(grid_start, grid_end, blksize)):
            coords = coords_device[ip0:ip1]
            weight = weights_device[ip0:ip1]
            # cache ao indices
            lookup_key = (device_id, block_id, blksize, ngrids)
            if lookup_key not in ni.non0ao_idx:
                ni.non0ao_idx[lookup_key] = _sparse_index(_sorted_mol, coords, opt.l_ctr_offsets)
            
            pad, idx, non0shl_idx, ctr_offsets_slice, ao_loc_slice = ni.non0ao_idx[lookup_key]
            ao_mask = eval_ao(
                _sorted_mol, coords, deriv,
                nao_slice=len(idx),
                shls_slice=non0shl_idx,
                ao_loc_slice=ao_loc_slice,
                ctr_offsets_slice=ctr_offsets_slice,
                gdftopt=opt,
                transpose=False
            )
            
            if pad > 0:
                if deriv == 0:
                    ao_mask[-pad:,:] = 0.0
                else:
                    ao_mask[:,-pad:,:] = 0.0
            yield ao_mask, idx, weight, coords

def _grouped_block_loop(ni, mol, grids, nao=None, deriv=0, max_memory=2000,
                non0tab=None, blksize=None, buf=None, extra=0):
    '''
    Define this macro to loop over grids by blocks.
    Sparsity is not implemented yet
    sorted_ao: by default ao_value is sorted for GPU
    '''
    if grids.coords is None:
        grids.build(with_non0tab=False, sort_grids=True)
    if nao is None:
        nao = mol.nao
    ngrids = grids.coords.shape[0]
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    log = logger.new_logger(mol, mol.verbose)

    if blksize is None:
        #cupy.get_default_memory_pool().free_all_blocks()
        mem_avail = get_avail_mem()
        blksize = int((mem_avail*.2/8/((comp+1)*nao + extra))/ ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, blksize)
        if blksize < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    ao_mask_group = []
    idx_group = []
    weight_group = []
    coords_group = []
    total_used_bytes = 0
    mem_limit = get_avail_mem()

    _sorted_mol = opt._sorted_mol
    with opt.gdft_envs_cache():
        block_id = 0
        t1 = log.init_timer()
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            # cache ao indices
            if (block_id, blksize, ngrids) not in ni.non0ao_idx:
                ni.non0ao_idx[block_id, blksize, ngrids] = _sparse_index(_sorted_mol, coords, opt.l_ctr_offsets)

            pad, idx, non0shl_idx, ctr_offsets_slice, ao_loc_slice = ni.non0ao_idx[block_id, blksize, ngrids]

            ao_mask = eval_ao(
                _sorted_mol, coords, deriv,
                nao_slice=len(idx),
                shls_slice=non0shl_idx,
                ao_loc_slice=ao_loc_slice,
                ctr_offsets_slice=ctr_offsets_slice,
                gdftopt=opt,
                transpose=False
            )

            if pad > 0:
                if deriv == 0:
                    ao_mask[-pad:,:] = 0.0
                else:
                    ao_mask[:,-pad:,:] = 0.0
            block_id += 1
            total_used_bytes += ao_mask.nbytes
            ao_mask_group.append(ao_mask)
            idx_group.append(idx)
            weight_group.append(weight)
            coords_group.append(coords)
            if total_used_bytes > 0.2 * mem_limit:
                t1 = log.timer_debug2('evaluate ao slice', *t1)
                yield ao_mask_group, idx_group, weight_group, coords_group
                ao_mask_group = []
                idx_group = []
                weight_group = []
                coords_group = []
                total_used_bytes = 0
        if total_used_bytes > 0:
            t1 = log.timer_debug2('evaluate ao slice', *t1)
            yield ao_mask_group, idx_group, weight_group, coords_group

class LibXCMixin:
    libxc = libxc
    omega = None
    to_cpu = NotImplemented

    def hybrid_coeff(self, xc_code, spin=0):
        return dft.libxc.hybrid_coeff(xc_code, spin)

    def nlc_coeff(self, xc_code):
        return dft.libxc.nlc_coeff(xc_code)

    def rsh_coeff(sef, xc_code):
        return dft.libxc.rsh_coeff(xc_code)
    eval_xc      = NotImplemented
    eval_xc_eff  = NotImplemented

    def _xc_type(self, xc_code):
        return dft.libxc.xc_type(xc_code)

    rsh_and_hybrid_coeff = numint.LibXCMixin.rsh_and_hybrid_coeff

_NumIntMixin = LibXCMixin
from gpu4pyscf.lib import utils
class NumInt(lib.StreamObject, LibXCMixin):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'screen_index', 'xcfuns', 'gdftopt', 'pair_mask', 'grid_blksize', 'non0ao_idx'}
    gdftopt      = None
    pair_mask    = None
    screen_index = None
    xcfuns       = None        # can be multiple xc functionals

    def build(self, mol, coords):
        self.gdftopt = _GDFTOpt.from_mol(mol)
        self.grid_blksize = None
        self.non0ao_idx = {}
        return self

    get_rho = get_rho
    nr_rks = nr_rks
    nr_uks = nr_uks
    nr_nlc_vxc = nr_nlc_vxc
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    cache_xc_kernel = cache_xc_kernel

    # cannot patch this function
    eval_xc_eff = eval_xc_eff
    block_loop = _block_loop
    eval_ao = staticmethod(eval_ao)
    eval_rho = staticmethod(eval_rho)
    eval_rho2 = staticmethod(eval_rho2)

    def to_cpu(self):
        ni = numint.NumInt()
        return ni

    @lru_cache(10)
    def _init_xcfuns(self, xc_code, spin=0):
        return _init_xcfuns(xc_code, spin)

    def reset(self):
        self.gdftopt      = None
        self.pair_mask    = None
        self.screen_index = None
        self.xcfuns       = None
        self.grid_blksize = None
        self.non0ao_idx = {}
        return self

def _make_pairs2shls_idx(pair_mask, l_bas_loc, hermi=0):
    if hermi:
        pair_mask = np.tril(pair_mask)
    locs = l_bas_loc // BAS_ALIGNED
    assert locs[-1] == pair_mask.shape[0]
    pair2bra = []
    pair2ket = []
    for i0, i1 in zip(locs[:-1], locs[1:]):
        for j0, j1 in zip(locs[:-1], locs[1:]):
            idx, idy = np.where(pair_mask[i0:i1,j0:j1])
            pair2bra.append((i0 + idx) * BAS_ALIGNED)
            pair2ket.append((j0 + idy) * BAS_ALIGNED)
            if hermi and i0 == j0:
                break
    bas_pairs_locs = np.append(
            0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
    bas_pair2shls = np.hstack(
            pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
    return bas_pair2shls, bas_pairs_locs

def _contract_rho(bra, ket, rho=None):
    if bra.flags.c_contiguous and ket.flags.c_contiguous:
        assert bra.shape == ket.shape
        nao, ngrids = bra.shape
        if rho is None:
            rho = cupy.empty(ngrids)
        stream = cupy.cuda.get_current_stream()
        err = libgdft.GDFTcontract_rho(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(rho.data.ptr, ctypes.c_void_p),
            ctypes.cast(bra.data.ptr, ctypes.c_void_p),
            ctypes.cast(ket.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids), ctypes.c_int(nao))
        if err != 0:
            raise RuntimeError('CUDA Error')
    else:
        rho = contract('ig,ig->g', bra, ket)
    return rho

def _contract_rho1(bra, ket, rho=None):
    ''' xip,ip->xp
    '''
    if bra.ndim == 2:
        bra = cupy.expand_dims(bra, axis=0)
    nvar, nao, ngrids = bra.shape
    if rho is None:
        rho = cupy.empty([nvar, ngrids])

    for i in range(nvar):
        stream = cupy.cuda.get_current_stream()
        err = libgdft.GDFTcontract_rho(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(rho[i].data.ptr, ctypes.c_void_p),
            ctypes.cast(bra[i].data.ptr, ctypes.c_void_p),
            ctypes.cast(ket.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids), ctypes.c_int(nao))
        if err != 0:
            raise RuntimeError('CUDA Error')
    return rho

def _contract_rho_gga(bra, ket, rho=None):
    ''' ig,nig->ng
    '''
    n, nao, ngrids = bra.shape
    assert n == 4
    if rho is None:
        rho = cupy.empty([4,ngrids])
    stream = cupy.cuda.get_current_stream()
    err = libgdft.GDFTcontract_rho_gga(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(rho.data.ptr, ctypes.c_void_p),
        ctypes.cast(bra.data.ptr, ctypes.c_void_p),
        ctypes.cast(ket.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids), ctypes.c_int(nao))
    if err != 0:
        raise RuntimeError('CUDA Error')
    return rho

def _contract_rho_mgga(bra, ket, rho=None):
    ''' nig,nig->ng
    '''
    n, nao, ngrids = bra.shape
    assert n == 4
    if rho is None:
        rho = cupy.empty([5,ngrids])
    stream = cupy.cuda.get_current_stream()
    err = libgdft.GDFTcontract_rho_mgga(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(rho.data.ptr, ctypes.c_void_p),
        ctypes.cast(bra.data.ptr, ctypes.c_void_p),
        ctypes.cast(ket.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids), ctypes.c_int(nao))
    if err != 0:
        raise RuntimeError('CUDA Error')
    return rho

def _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    return cupy.dot(dm.T, ao)

def _dot_ao_ao(mol, ao1, ao2, non0tab, shls_slice, ao_loc, hermi=0):
    return cupy.dot(ao1, ao2.T)


def _dot_ao_dm_sparse(ao, dm, nbins, screen_index, pair_mask, ao_loc,
                      l_bas_offsets):
    assert ao.flags.f_contiguous
    assert ao.dtype == dm.dtype == np.double
    ngrids, nao = ao.shape
    nbas = ao_loc.size - 1
    nsegs = l_bas_offsets.size - 1
    out = cupy.empty((nao, ngrids)).T
    err = libgdft.GDFTdot_ao_dm_sparse(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.c_int(dm.flags.c_contiguous),
        ctypes.c_int(ngrids), ctypes.c_int(nbas),
        ctypes.c_int(nbins), ctypes.c_int(nsegs),
        l_bas_offsets.ctypes.data_as(ctypes.c_void_p),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        pair_mask.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('CUDA Error')
    return out

def _dot_ao_ao_sparse(bra, ket, wv, nbins, screen_index, ao_loc,
                      bas_pair2shls, bas_pairs_locs, out):
    assert bra.flags.c_contiguous
    assert ket.flags.c_contiguous
    assert bra.dtype == ket.dtype == np.double
    nao, ngrids = bra.shape
    nbas = ao_loc.size - 1
    npair_segs = bas_pairs_locs.size - 1
    if wv is None:
        err = libgdft.GDFTdot_ao_ao_sparse(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(bra.data.ptr, ctypes.c_void_p),
            ctypes.cast(ket.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids), ctypes.c_int(nbas),
            ctypes.c_int(nbins), ctypes.c_int(npair_segs),
            bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
            bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            screen_index.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
    else:
        err = libgdft.GDFTdot_aow_ao_sparse(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(bra.data.ptr, ctypes.c_void_p),
            ctypes.cast(ket.data.ptr, ctypes.c_void_p),
            ctypes.cast(wv.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids), ctypes.c_int(nbas),
            ctypes.c_int(nbins), ctypes.c_int(npair_segs),
            bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
            bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            screen_index.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))

    if err != 0:
        raise RuntimeError('CUDA Error')
    return out

def _tau_dot_sparse(bra, ket, wv, nbins, screen_index, ao_loc,
                    bas_pair2shls, bas_pairs_locs, out):
    '''1/2 <nabla i| v | nabla j>'''
    wv = .5 * wv
    _dot_ao_ao_sparse(bra[1], ket[1], wv, nbins, screen_index,
                      ao_loc, bas_pair2shls, bas_pairs_locs, out)
    _dot_ao_ao_sparse(bra[2], ket[2], wv, nbins, screen_index,
                      ao_loc, bas_pair2shls, bas_pairs_locs, out)
    _dot_ao_ao_sparse(bra[3], ket[3], wv, nbins, screen_index,
                      ao_loc, bas_pair2shls, bas_pairs_locs, out)
    return out

def _scale_ao(ao, wv, out=None):
    if wv.ndim == 1:
        if ao.flags.f_contiguous:
            return ao * wv
        nvar = 1
        nao, ngrids = ao.shape
        assert wv.size == ngrids
    else:
        if ao[0].flags.f_contiguous:
            return contract('nip,np->ip', ao, wv)
        nvar, nao, ngrids = ao.shape
        assert wv.shape == (nvar, ngrids)

    wv = cupy.asarray(wv, order='C')
    if out is None:
        out = cupy.empty((nao, ngrids), order='C')
    stream = cupy.cuda.get_current_stream()
    err = libgdft.GDFTscale_ao(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao.data.ptr, ctypes.c_void_p),
        ctypes.cast(wv.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids), ctypes.c_int(nao), ctypes.c_int(nvar))
    if err != 0:
        raise RuntimeError('CUDA Error')
    return out

def _tau_dot(bra, ket, wv):
    '''1/2 <nabla i| v | nabla j>'''
    wv = cupy.asarray(.5 * wv)
    mat  = bra[1].dot(_scale_ao(ket[1], wv).T)
    mat += bra[2].dot(_scale_ao(ket[2], wv).T)
    mat += bra[3].dot(_scale_ao(ket[3], wv).T)
    return mat

class _GDFTOpt:
    def __init__(self, mol):
        self.envs_cache = {}
        self._sorted_mol = None       # sorted mol object based on contraction pattern
        self.mol = mol

    def build(self, mol=None):
        if mol is None:
            mol = self.mol
        else:
            self.mol = mol
        if hasattr(mol, '_decontracted') and mol._decontracted:
            raise RuntimeError('mol object is already decontracted')

        pmol = basis_seg_contraction(mol, allow_replica=True)
        pmol.cart = mol.cart
        coeff = cupy.eye(mol.nao)      # without cart2sph transformation
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

        # Paddings to make basis aligned in each angular momentum group
        inv_idx_padding = []
        l_counts = []
        bas_to_pad = []
        for l in range(LMAX_ON_GPU+1):
            l_count = l_ctr_counts[uniq_l_ctr[:,0] == l].sum()
            if l_count == 0:
                continue
            padding_len = (-l_count) % BAS_ALIGNED
            if padding_len > 0:
                logger.debug(mol, 'Padding %d basis for l=%d', padding_len, l)
                l_ctr_type = np.where(uniq_l_ctr[:,0] == l)[0][-1]
                l_ctr_counts[l_ctr_type] += padding_len
                bas_idx_dup = np.where(inv_idx == l_ctr_type)[0][-1]
                bas_to_pad.extend([bas_idx_dup] * padding_len)
                inv_idx_padding.extend([l_ctr_type] * padding_len)

            l_counts.append(l_count + padding_len)

        # Padding inv_idx, pmol._bas
        if inv_idx_padding:
            inv_idx = np.append(inv_idx, inv_idx_padding)
            pmol._bas = np.vstack([pmol._bas, pmol._bas[bas_to_pad]])

        ao_loc = pmol.ao_loc_nr()
        nao = ao_loc[-1]
        sorted_idx = np.argsort(inv_idx.ravel())
        pmol._bas = np.asarray(pmol._bas[sorted_idx], dtype=np.int32)
        ao_idx = np.array_split(np.arange(nao), ao_loc[1:-1])
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        assert pmol.nbas % BAS_ALIGNED == 0
        # Padding zeros to transformation coefficients
        if nao > coeff.shape[0]:
            paddings = nao - coeff.shape[0]
            coeff = np.vstack([coeff, np.zeros((paddings, coeff.shape[1]))])
        pmol._decontracted = True
        self._sorted_mol = pmol
        self._ao_idx = np.asarray(ao_idx, dtype=np.int32)
        self.coeff = coeff[ao_idx]
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts)).astype(np.int32)
        self.l_bas_offsets = np.append(0, np.cumsum(l_counts)).astype(np.int32)
        logger.debug2(mol, 'l_ctr_offsets = %s', self.l_ctr_offsets)
        logger.debug2(mol, 'l_bas_offsets = %s', self.l_bas_offsets)
        return self

    @classmethod
    def from_mol(cls, mol):
        return cls(mol).build()

    @contextlib.contextmanager
    def gdft_envs_cache(self):
        _sorted_mol = self._sorted_mol
        ao_loc = _sorted_mol.ao_loc_nr()
        device_id = cupy.cuda.Device().id
        envs_cache = ctypes.POINTER(_GDFTEnvsCache)()
        libgdft.GDFTinit_envs(
            ctypes.byref(envs_cache), ao_loc.ctypes.data_as(ctypes.c_void_p),
            _sorted_mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol.natm),
            _sorted_mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol.nbas),
            _sorted_mol._env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol._env.size))
        self.envs_cache[device_id] = envs_cache
        try:
            yield
        finally:
            envs_cache = self.envs_cache[device_id]
            libgdft.GDFTdel_envs(ctypes.byref(envs_cache))

    def sort_orbitals(self, mat, axis=[]):
        ''' Transform given axis of a matrix into sorted AO
        '''
        idx = self._ao_idx
        shape_ones = (1,) * mat.ndim
        fancy_index = []
        for dim, n in enumerate(mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[], out=None):
        ''' Transform given axis of a matrix into original AO
        '''
        idx = self._ao_idx
        shape_ones = (1,) * sorted_mat.ndim
        fancy_index = []
        for dim, n in enumerate(sorted_mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        if out is None:
            out = cupy.empty_like(sorted_mat)
        out[tuple(fancy_index)] = sorted_mat
        return out

class _GDFTEnvsCache(ctypes.Structure):
    pass
