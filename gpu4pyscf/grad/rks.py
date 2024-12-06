#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

'''Non-relativistic RKS analytical nuclear gradients'''
from concurrent.futures import ThreadPoolExecutor
import ctypes
import numpy
import cupy
from pyscf import lib, gto
from pyscf.grad import rks as rks_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.dft import numint, xc_deriv
from gpu4pyscf.dft import radi
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, add_sparse, tag_array, sandwich_dot, reduce_to_device)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, _num_devices

from pyscf import __config__
MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 128*128)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)

libgdft = numint.libgdft
libgdft.GDFT_make_dR_dao_w.restype = ctypes.c_int

def get_veff(ks_grad, mol=None, dm=None, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    Veff per atom.

    NOTE: This function is incompatible to the one implemented in PySCF CPU version.
    In the CPU version, get_veff returns the first order derivatives of Veff matrix.

    Args:
        ks_grad : grad.rhf.Gradients or grad.rks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(sort_grids=True)

    nlcgrids = None
    if mf.do_nlc():
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(sort_grids=True)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            raise NotImplementedError
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    # this can be moved into vxc calculations
    occ_coeff = cupy.asarray(mf.mo_coeff[:, mf.mo_occ>0.5], order='C')
    tmp = contract('nij,jk->nik', vxc, occ_coeff)
    exc1_per_atom = 2.0*contract('nik,ik->ni', tmp, occ_coeff)

    aoslices = mol.aoslice_by_atom()
    exc1_per_atom = [exc1_per_atom[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    exc1_per_atom = cupy.asarray(exc1_per_atom)

    vhfopt = mf._opt_gpu.get(None, None)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        ej = rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, with_k=False,
                                          verbose=verbose)[0]
        exc1_per_atom += ej
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        ej, ek = rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, verbose=verbose)
        ek *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            vhfopt = mf._opt_gpu.get(omega, None)
            with mol.with_range_coulomb(omega):
                ek_lr = rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, with_j=False,
                                                     verbose=verbose)[1]
            ek += ek_lr * (alpha - hyb)
        exc1_per_atom += ej - ek * .5
    return tag_array(exc1_per_atom, exc1_grid=exc)

def _get_vxc_task(ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                  verbose=None, with_lapl=False, grid_range=(), device_id=0):
    ''' Calculate the gradient of vxc on given device
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
        ngrids_glob = grids.coords.shape[0]
        ngrids_per_device = (ngrids_glob + _num_devices - 1) // _num_devices
        grid_start = device_id * ngrids_per_device
        grid_end = (device_id + 1) * ngrids_per_device
        
        nset = len(dms)
        assert nset == 1
        vmat = cupy.zeros((nset,3,nao,nao))
        if xctype == 'LDA':
            ao_deriv = 1
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None, 
                                                         grid_range=(grid_start, grid_end)):
                for idm in range(nset):
                    mo_coeff_mask = mo_coeff[idx,:]
                    rho = numint.eval_rho2(_sorted_mol, ao_mask[0], mo_coeff_mask, mo_occ, None, xctype)
                    vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                    wv = weight * vxc[0]
                    aow = numint._scale_ao(ao_mask[0], wv)
                    vtmp = _d1_dot_(ao_mask[1:4], aow.T)
                    add_sparse(vmat[idm], vtmp, idx)
        elif xctype == 'GGA':
            ao_deriv = 2
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                for idm in range(nset):
                    mo_coeff_mask = mo_coeff[idx,:]
                    rho = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ, None, xctype)
                    vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                    wv = weight * vxc
                    wv[0] *= .5
                    vtmp = _gga_grad_sum_(ao_mask, wv)
                    add_sparse(vmat[idm], vtmp, idx)
        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 2
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                for idm in range(nset):
                    mo_coeff_mask = mo_coeff[idx,:]
                    rho = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ, None, xctype, with_lapl=False)
                    vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                    wv = weight * vxc
                    wv[0] *= .5
                    wv[4] *= .5  # for the factor 1/2 in tau
                    vtmp = _gga_grad_sum_(ao_mask, wv)
                    vtmp += _tau_grad_dot_(ao_mask, wv[4])
                    add_sparse(vmat[idm], vtmp, idx)
        log.timer_debug1('gradient of vxc', *t0)
    return vmat

def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    
    mo_occ = cupy.asarray(dms.mo_occ)
    mo_coeff = cupy.asarray(dms.mo_coeff)
    nao = mol.nao
    dms = cupy.asarray(dms).reshape(-1,nao,nao)
    nset = dms.shape[0]
    dms = opt.sort_orbitals(dms, axis=[1,2])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _get_vxc_task,
                ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                verbose=verbose, device_id=device_id)
            futures.append(future)
    vmat_dist = [future.result() for future in futures]
    vmat = reduce_to_device(vmat_dist)
    vmat = opt.unsort_orbitals(vmat, axis=[2,3])
    exc = None
    if nset == 1:
        vmat = vmat[0]

    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def get_nlc_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_occ = cupy.asarray(dms.mo_occ)
    mo_coeff = cupy.asarray(dms.mo_coeff)

    mol = None
    _sorted_mol = opt._sorted_mol
    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms).reshape(-1,nao0,nao0)
    dms = opt.sort_orbitals(dms, axis=[1,2])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    nset = len(dms)
    assert nset == 1

    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    ao_deriv = 2
    vvrho = []
    for ao_mask, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=max_memory):
        mo_coeff_mask = mo_coeff[mask]
        rho = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ, None, xctype, with_lapl=False)
        vvrho.append(rho)
    rho = cupy.hstack(vvrho)

    vxc = numint._vv10nlc(rho, grids.coords, rho, grids.weights,
                          grids.coords, nlc_pars)[1]
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    vmat = cupy.zeros((3,nao,nao))
    p1 = 0
    for ao_mask, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5  # *.5 because vmat + vmat.T at the end
        vmat_tmp = _gga_grad_sum_(ao_mask, wv)
        add_sparse(vmat, vmat_tmp, mask)

    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    exc = None
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def _make_dR_dao_w(ao, wv):
    #:aow = numpy.einsum('npi,p->npi', ao[1:4], wv[0])
    '''
    aow = [
        numint._scale_ao(ao[1], wv[0]),  # dX nabla_x
        numint._scale_ao(ao[2], wv[0]),  # dX nabla_y
        numint._scale_ao(ao[3], wv[0]),  # dX nabla_z
    ]
    # XX, XY, XZ = 4, 5, 6
    # YX, YY, YZ = 5, 7, 8
    # ZX, ZY, ZZ = 6, 8, 9
    aow[0] += numint._scale_ao(ao[4], wv[1])  # dX nabla_x
    aow[0] += numint._scale_ao(ao[5], wv[2])  # dX nabla_y
    aow[0] += numint._scale_ao(ao[6], wv[3])  # dX nabla_z
    aow[1] += numint._scale_ao(ao[5], wv[1])  # dY nabla_x
    aow[1] += numint._scale_ao(ao[7], wv[2])  # dY nabla_y
    aow[1] += numint._scale_ao(ao[8], wv[3])  # dY nabla_z
    aow[2] += numint._scale_ao(ao[6], wv[1])  # dZ nabla_x
    aow[2] += numint._scale_ao(ao[8], wv[2])  # dZ nabla_y
    aow[2] += numint._scale_ao(ao[9], wv[3])  # dZ nabla_z
    '''
    assert ao.flags.c_contiguous
    assert wv.flags.c_contiguous

    _, nao, ngrids = ao.shape
    aow = cupy.empty([3,nao,ngrids])
    stream = cupy.cuda.get_current_stream()
    err = libgdft.GDFT_make_dR_dao_w(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(aow.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao.data.ptr, ctypes.c_void_p),
        ctypes.cast(wv.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids), ctypes.c_int(nao))
    if err != 0:
        raise RuntimeError('CUDA Error')
    return aow

def _d1_dot_(ao1, ao2, out=None):
    if out is None:
        out = cupy.empty([3, ao1[0].shape[0], ao2.shape[1]])
        out[0] = cupy.dot(ao1[0], ao2)
        out[1] = cupy.dot(ao1[1], ao2)
        out[2] = cupy.dot(ao1[2], ao2)
        return out
        #return cupy.stack([vmat0,vmat1,vmat2])
    else:
        cupy.dot(ao1[0], ao2, out=out[0])
        cupy.dot(ao1[1], ao2, out=out[1])
        cupy.dot(ao1[2], ao2, out=out[2])
        return out

def _gga_grad_sum_(ao, wv):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
    aow = numint._scale_ao(ao[:4], wv[:4])
    vmat = _d1_dot_(ao[1:4], aow.T)
    aow = _make_dR_dao_w(ao, wv[:4])
    vmat += _d1_dot_(aow, ao[0].T)
    return vmat

# XX, XY, XZ = 4, 5, 6
# YX, YY, YZ = 5, 7, 8
# ZX, ZY, ZZ = 6, 8, 9
def _tau_grad_dot_(ao, wv):
    '''The tau part of MGGA functional'''
    aow = numint._scale_ao(ao[1], wv)
    vmat = _d1_dot_([ao[4], ao[5], ao[6]], aow.T)
    aow = numint._scale_ao(ao[2], wv)
    vmat += _d1_dot_([ao[5], ao[7], ao[8]], aow.T)
    aow = numint._scale_ao(ao[3], wv)
    vmat += _d1_dot_([ao[6], ao[8], ao[9]], aow.T)
    return vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    log = logger.new_logger(mol, verbose)
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    natm = mol.natm
    mol = None
    _sorted_mol = opt._sorted_mol
    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms)
    assert dms.ndim == 2
    #:dms = cupy.einsum('pi,ij,qj->pq', coeff, dms, coeff)
    dms = sandwich_dot(dms, coeff.T)

    excsum = cupy.zeros((natm, 3))
    vmat = cupy.zeros((3,nao,nao))
    with opt.gdft_envs_cache():
        if xctype == 'LDA':
            ao_deriv = 1
        else:
            ao_deriv = 2

        mem_avail = get_avail_mem()
        comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
        block_size = int((mem_avail*.4/8/(comp+1)/nao - 3*nao*2)/ ALIGNED) * ALIGNED
        block_size = min(block_size, MIN_BLK_SIZE)
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)

        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            ngrids = weight.size
            for p0, p1 in lib.prange(0,ngrids,block_size):
                ao = numint.eval_ao(_sorted_mol, coords[p0:p1, :], ao_deriv, gdftopt=opt, transpose=False)

                if xctype == 'LDA':
                    rho = numint.eval_rho(_sorted_mol, ao[0], dms,
                                          xctype=xctype, hermi=1, with_lapl=False)
                    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
                    exc = exc[:,0]
                    wv = weight[p0:p1] * vxc[0]
                    aow = numint._scale_ao(ao[0], wv)
                    vtmp = _d1_dot_(ao[1:4], aow.T)
                    vmat += vtmp
                    # response of weights
                    excsum += cupy.einsum('r,nxr->nx', exc*rho, weight1[:,:,p0:p1])
                    # response of grids coordinates
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms) * 2
                    rho = vxc = aow = None

                elif xctype == 'GGA':
                    rho = numint.eval_rho(_sorted_mol, ao[:4], dms,
                                          xctype=xctype, hermi=1, with_lapl=False)
                    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
                    exc = exc[:,0]
                    wv = weight[p0:p1] * vxc
                    wv[0] *= .5
                    vtmp = _gga_grad_sum_(ao, wv)
                    vmat += vtmp
                    excsum += cupy.einsum('r,nxr->nx', exc*rho[0], weight1[:,:,p0:p1])
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms) * 2
                    rho = vxc = None

                elif xctype == 'NLC':
                    raise NotImplementedError

                elif xctype == 'MGGA':
                    rho = numint.eval_rho(_sorted_mol, ao, dms,
                                          xctype=xctype, hermi=1, with_lapl=False)
                    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
                    exc = exc[:,0]
                    wv = weight[p0:p1] * vxc
                    wv[0] *= .5
                    wv[4] *= .5  # for the factor 1/2 in tau

                    vtmp  = _gga_grad_sum_(ao, wv)
                    vtmp += _tau_grad_dot_(ao, wv[4])
                    vmat += vtmp
                    excsum += cupy.einsum('r,nxr->nx', exc*rho[0], weight1[:,:,p0:p1])
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms) * 2
                    rho = vxc = None

    #:vmat = cupy.einsum('pi,npq,qj->nij', coeff, vmat, coeff)
    vmat = sandwich_dot(vmat, coeff)

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


# JCP 98, 5612 (1993); DOI:10.1063/1.464906
def grids_response_cc(grids):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.inter_distance(mol, atm_coords)
    atm_dist = cupy.asarray(atm_dist)
    atm_coords = cupy.asarray(atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = []
        grid_norm_vec = []
        for ia in range(mol.natm):
            v = (atm_coords[ia] - coords).T
            normv = numpy.linalg.norm(v,axis=0) + 1e-200
            v /= normv
            grid_dist.append(normv)
            grid_norm_vec.append(v)

        def get_du(ia, ib):  # JCP 98, 5612 (1993); (B10)
            uab = atm_coords[ia] - atm_coords[ib]
            duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
            duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
            return duab

        pbecke = cupy.ones((mol.natm,ngrids))
        dpbecke = cupy.zeros((mol.natm,mol.natm,3,ngrids))
        for ia in range(mol.natm):
            for ib in range(ia):
                g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                p0 = fadjust(ia, ib, g)
                p1 = (3 - p0**2) * p0 * .5
                p2 = (3 - p1**2) * p1 * .5
                p3 = (3 - p2**2) * p2 * .5
                t_uab = 27./16 * (1-p2**2) * (1-p1**2) * (1-p0**2) * gadjust(ia, ib, g)

                s_uab = .5 * (1 - p3 + 1e-200)
                s_uba = .5 * (1 + p3 + 1e-200)

                pbecke[ia] *= s_uab
                pbecke[ib] *= s_uba
                pt_uab =-t_uab / s_uab
                pt_uba = t_uab / s_uba

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
                duab = get_du(ia, ib)
                duba = get_du(ib, ia)
                if ia == atom_id:
                    dpbecke[ia,ia] += pt_uab * duba
                    dpbecke[ia,ib] += pt_uba * duba
                else:
                    dpbecke[ia,ia] += pt_uab * duab
                    dpbecke[ia,ib] += pt_uba * duab

                if ib == atom_id:
                    dpbecke[ib,ib] -= pt_uba * duab
                    dpbecke[ib,ia] -= pt_uab * duab
                else:
                    dpbecke[ib,ib] -= pt_uba * duba
                    dpbecke[ib,ia] -= pt_uab * duba

# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                if ia != atom_id and ib != atom_id:
                    ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                    ua_ub /= atm_dist[ia,ib]
                    dpbecke[atom_id,ia] -= pt_uab * ua_ub
                    dpbecke[atom_id,ib] -= pt_uba * ua_ub

        for ia in range(mol.natm):
            dpbecke[:,ia] *= pbecke[ia]
        return pbecke, dpbecke

    natm = mol.natm
    for ia in range(natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = cupy.asarray(coords)
        vol = cupy.asarray(vol)

        coords = coords + cupy.asarray(atm_coords[ia])
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = 1./pbecke.sum(axis=0)
        w1 = dpbecke[:,ia] * z
        w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
        w1 *= vol
        w0 = vol * pbecke[ia] * z
        yield coords, w0, w1

class Gradients(rhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device
    # attributes
    grid_response = False
    _keys = rks_grad.Gradients._keys

    def __init__ (self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        if self.grid_response:
            vhf = envs['dvhf']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            return vhf.exc1_grid[atom_id]
        else:
            return 0

Grad = Gradients
from gpu4pyscf import dft
dft.rks.RKS.Gradients = lib.class_as_method(Gradients)
