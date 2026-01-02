# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, add_sparse, tag_array, sandwich_dot,
    reduce_to_device, take_last2d, ndarray)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
from gpu4pyscf.gto.mole import groupby, ATOM_OF

from pyscf import __config__
MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 4096)
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
    if not hasattr(dm, "mo_coeff"): dm = tag_array(dm, mo_coeff = ks_grad.base.mo_coeff)
    if not hasattr(dm, "mo_occ"):   dm = tag_array(dm,   mo_occ = ks_grad.base.mo_occ)
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(sort_grids=True)

    if ks_grad.grid_response:
        exc, exc1 = get_exc_full_response(ni, mol, grids, mf.xc, dm, verbose=log)
        exc1 += exc/2
    else:
        exc, exc1 = get_exc(ni, mol, grids, mf.xc, dm, verbose=log)
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if mf.do_nlc():
        enlc1_per_atom, enlc1_grid = _get_denlc(ks_grad, mol, dm)
        exc1 += enlc1_per_atom
        if ks_grad.grid_response:
            exc1 += enlc1_grid/2

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    vhfopt = mf._opt_gpu.get(mol.omega)
    j_factor = 1.
    k_factor = 0.
    if with_k:
        if omega == 0:
            k_factor = hyb
        elif alpha == 0: # LR=0, only SR exchange
            pass
        elif hyb == 0: # SR=0, only LR exchange
            k_factor = alpha
        else: # SR and LR exchange with different ratios
            k_factor = alpha
    exc1 += rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, j_factor, k_factor,
                                         verbose=log) * .5
    if with_k and omega != 0:
        j_factor = 0.
        omega = -omega # Prefer computing the SR part
        if alpha == 0: # LR=0, only SR exchange
            k_factor = hyb
        elif hyb == 0: # SR=0, only LR exchange
            # full range exchange was computed in the previous step
            k_factor = -alpha
        else: # SR and LR exchange with different ratios
            k_factor = hyb - alpha # =beta
        vhfopt = mf._opt_gpu.get(omega)
        with mol.with_range_coulomb(omega):
            exc1 += rhf_grad._jk_energy_per_atom(
                mol, dm, vhfopt, j_factor, k_factor, verbose=log) * .5
    return exc1

def _get_denlc(ks_grad, mol, dm):
    mf = ks_grad.base
    ni = mf._numint
    assert mf.do_nlc()

    if ks_grad.nlcgrids is not None:
        nlcgrids = ks_grad.nlcgrids
    else:
        nlcgrids = mf.nlcgrids
    if nlcgrids.coords is None:
        nlcgrids.build(sort_grids=True)

    if ni.libxc.is_nlc(mf.xc):
        xc = mf.xc
    else:
        xc = mf.nlc

    if ks_grad.grid_response:
        enlc, enlc1_per_atom = get_nlc_exc_full_response(
            ni, mol, nlcgrids, xc, dm, verbose=ks_grad.verbose)
    else:
        enlc, enlc1_per_atom = get_nlc_exc(
            ni, mol, nlcgrids, xc, dm, verbose=ks_grad.verbose)
    return enlc1_per_atom, enlc

def _get_exc_task(ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                  verbose=None, with_lapl=False, device_id=0):
    ''' Calculate the gradient of vxc on given device
    '''
    with cupy.cuda.Device(device_id):
        if dms is not None: dms = cupy.asarray(dms)
        if mo_coeff is not None: mo_coeff = cupy.asarray(mo_coeff)
        if mo_occ is not None: mo_occ = cupy.asarray(mo_occ)
        dm, dms = dms[0], None

        log = logger.new_logger(mol, verbose)
        t0 = log.init_timer()
        xctype = ni._xc_type(xc_code)
        nao = mol.nao
        opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol
        nocc = cupy.count_nonzero(mo_occ>0)

        ngrids_glob = grids.coords.shape[0]
        grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
        ngrids_local = grid_end - grid_start
        log.debug(f"{ngrids_local} grids on Device {device_id}")

        exc1_ao = cupy.zeros((nao,3))
        vtmp_buf = cupy.empty((3*nao*nao))
        mo_buf = cupy.empty_like(mo_coeff)
        dm_mask_buf = cupy.empty(nao*nao)
        if xctype == 'LDA':
            ao_deriv = 1
            aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao, 1*nocc))
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                mo_coeff_mask = cupy.take(mo_coeff, idx, axis=0, out=mo_buf[:len(idx)])
                rho = numint.eval_rho2(_sorted_mol, ao_mask[0], mo_coeff_mask,
                                       mo_occ, None, xctype, buf=aow_buf)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1][0]
                wv = cupy.multiply(weight, vxc, out=vxc)
                aow = numint._scale_ao(ao_mask[0], wv, out=aow_buf)
                vtmp = _d1_dot_(ao_mask[1:4], aow.T, out=vtmp_buf)
                dm_mask = take_last2d(dm, idx, out=dm_mask_buf)
                exc1_ao[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
        elif xctype == 'GGA':

            ao_deriv = 2
            aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao * 3, 2*nocc, 4))
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                mo_coeff_mask = cupy.take(mo_coeff, idx, axis=0, out=mo_buf[:len(idx)])
                rho = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask,
                                       mo_occ, None, xctype, buf=aow_buf)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, buf=aow_buf)[1]
                wv = cupy.multiply(weight, vxc, out=vxc)
                wv[0] *= .5
                vtmp = _gga_grad_sum_(ao_mask, wv, buf=aow_buf, out=vtmp_buf)
                dm_mask = take_last2d(dm, idx, out=dm_mask_buf)
                exc1_ao[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 2
            aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao * 3, 2*nocc, 5))
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                mo_coeff_mask = cupy.take(mo_coeff, idx, axis=0, out=mo_buf[:len(idx)])
                rho = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask,
                                       mo_occ, None, xctype, with_lapl=False, buf=aow_buf)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, buf=aow_buf)[1]
                wv = cupy.multiply(weight, vxc, out=vxc)
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                vtmp = _gga_grad_sum_(ao_mask, wv, buf=aow_buf, out=vtmp_buf)
                vtmp = _tau_grad_dot_(ao_mask, wv[4], accumulate=True, buf=aow_buf, out=vtmp)
                dm_mask = take_last2d(dm, idx, out=dm_mask_buf)
                exc1_ao[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T

        log.timer_debug1('gradient of vxc', *t0)
    return exc1_ao

def get_exc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_occ = cupy.asarray(dms.mo_occ)
    mo_coeff = cupy.asarray(dms.mo_coeff)
    nao = mol.nao
    dms = cupy.asarray(dms).reshape(-1,nao,nao)
    nset = dms.shape[0]
    assert nset == 1
    dms = opt.sort_orbitals(dms, axis=[1,2])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_exc_task,
                ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                verbose=log.verbose, device_id=device_id)
            futures.append(future)
    exc1_dist = [future.result() for future in futures]
    exc1 = reduce_to_device(exc1_dist)
    # - sign because nabla_X = -nabla_x
    exc1 = -_reduce_to_atom(opt._sorted_mol, exc1)
    log.timer_debug1('grad vxc', *t0)
    return None, exc1

def get_nlc_exc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_occ = cupy.asarray(dms.mo_occ)
    mo_coeff = cupy.asarray(dms.mo_coeff)

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    dms = cupy.asarray(dms).reshape(-1,nao,nao)
    dms = opt.sort_orbitals(dms, axis=[1,2])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    nset = len(dms)
    assert nset == 1
    dm, dms = dms[0], None

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

    exc1 = cupy.zeros((nao,3))
    p1 = 0
    for ao_mask, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5  # *.5 because vmat + vmat.T at the end
        vmat_tmp = _gga_grad_sum_(ao_mask, wv)
        #add_sparse(vmat, vmat_tmp, mask)
        dm_mask = dm[mask[:,None],mask]
        exc1[mask] += cupy.einsum('nij,ij->ni', vmat_tmp, dm_mask).T

    # - sign because nabla_X = -nabla_x
    exc1 = -_reduce_to_atom(opt._sorted_mol, exc1)
    log.timer_debug1('grad nlc vxc', *t0)
    return None, exc1

def _reduce_to_atom(mol, exc1):
    assert exc1.ndim == 2 and exc1.shape[1] == 3
    exc1 = cupy.asnumpy(exc1)
    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    atm_id_for_ao = numpy.repeat(mol._bas[:,ATOM_OF], dims)
    return groupby(atm_id_for_ao, exc1, op='sum')

def _make_dR_dao_w(ao, wv, out=None):
    #:aow = numpy.einsum('nip,p->nip', ao[1:4], wv[0])
    if not ao.flags.c_contiguous or ao.dtype != numpy.float64:
        aow = ndarray(ao[:3].shape, dtype=ao.dtype, buffer=out)
        tmp = cupy.empty_like(ao[0])
        numint._scale_ao(ao[1], wv[0], out=aow[0])  # dX nabla_x
        numint._scale_ao(ao[2], wv[0], out=aow[1])  # dX nabla_y
        numint._scale_ao(ao[3], wv[0], out=aow[2])  # dX nabla_z
        # XX, XY, XZ = 4, 5, 6
        # YX, YY, YZ = 5, 7, 8
        # ZX, ZY, ZZ = 6, 8, 9
        aow[0] += numint._scale_ao(ao[4], wv[1], out=tmp)  # dX nabla_x
        aow[0] += numint._scale_ao(ao[5], wv[2], out=tmp)  # dX nabla_y
        aow[0] += numint._scale_ao(ao[6], wv[3], out=tmp)  # dX nabla_z
        aow[1] += numint._scale_ao(ao[5], wv[1], out=tmp)  # dY nabla_x
        aow[1] += numint._scale_ao(ao[7], wv[2], out=tmp)  # dY nabla_y
        aow[1] += numint._scale_ao(ao[8], wv[3], out=tmp)  # dY nabla_z
        aow[2] += numint._scale_ao(ao[6], wv[1], out=tmp)  # dZ nabla_x
        aow[2] += numint._scale_ao(ao[8], wv[2], out=tmp)  # dZ nabla_y
        aow[2] += numint._scale_ao(ao[9], wv[3], out=tmp)  # dZ nabla_z
        return aow

    assert ao.flags.c_contiguous
    assert wv.flags.c_contiguous

    _, nao, ngrids = ao.shape
    aow = ndarray([3,nao,ngrids], buffer=out)
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

def _d1_dot_(ao1, ao2, alpha=1.0, beta=0.0, transpose=False, out=None):
    ao1 = cupy.asarray(ao1)
    ao2 = cupy.asarray(ao2)
    dtype = numpy.result_type(ao1, ao2)
    if not transpose:
        out = ndarray([3, ao1.shape[1], ao2.shape[1]], dtype=dtype, buffer=out)
        out = contract('bik,km->bim', ao1.conj(), ao2, alpha=alpha, beta=beta, out=out)
    else:
        out = ndarray([3, ao2.shape[1], ao1.shape[1]], dtype=dtype, buffer=out)
        out = contract('bik,km->bmi', ao1.conj(), ao2, alpha=alpha, beta=beta, out=out)
    return out

def _gga_grad_sum_(ao, wv, accumulate=False, buf=None, out=None):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
    buf = ndarray((3, ao.shape[1], ao.shape[2]), dtype=ao.dtype, buffer=buf)
    aow = numint._scale_ao(ao[:4], wv[:4], out=buf[0])
    if not accumulate:
        vmat = _d1_dot_(ao[1:4], aow.T, out=out)
    else:
        assert out is not None
        vmat = _d1_dot_(ao[1:4], aow.T, beta=1.0, out=out)
    aow = _make_dR_dao_w(ao, wv[:4], out=buf)
    vmat = _d1_dot_(aow, ao[0].T, beta=1, out=vmat)
    return vmat

# XX, XY, XZ = 4, 5, 6
# YX, YY, YZ = 5, 7, 8
# ZX, ZY, ZZ = 6, 8, 9
def _tau_grad_dot_(ao, wv, accumulate=False, buf=None, out=None):
    '''The tau part of MGGA functional'''
    idx1 = [4, 5, 6]
    idx2 = [5, 7, 8]
    idx3 = [6, 8, 9]
    aow = numint._scale_ao(ao[1], wv, out=buf)
    if accumulate:
        assert out is not None
        out = _d1_dot_(ao[idx1], aow.T, beta=1, out=out)
    else:
        out = _d1_dot_(ao[idx1], aow.T, out=out)
    aow = numint._scale_ao(ao[2], wv, out=aow)
    _d1_dot_(ao[idx2], aow.T, beta=1, out=out)
    aow = numint._scale_ao(ao[3], wv, out=aow)
    _d1_dot_(ao[idx3], aow.T, beta=1, out=out)
    return out

def get_exc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
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
    nao = _sorted_mol.nao
    dms = cupy.asarray(dms)
    assert dms.ndim == 2
    #:dms = cupy.einsum('pi,ij,qj->pq', coeff, dms, coeff)
    dms = opt.sort_orbitals(dms, axis=[0,1])

    excsum = cupy.zeros((natm, 3))
    vmat = cupy.zeros((3,nao,nao))

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
                _tau_grad_dot_(ao, wv[4], accumulate=True, out=vtmp)
                vmat += vtmp
                excsum += cupy.einsum('r,nxr->nx', exc*rho[0], weight1[:,:,p0:p1])
                excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms) * 2
                rho = vxc = None

    # - sign because nabla_X = -nabla_x
    exc1 = -.5 * rhf_grad.contract_h1e_dm(opt._sorted_mol, vmat, dms, hermi=1)
    return excsum.get(), exc1

def _vv10nlc_grad(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    # VV10 gradient term from Vydrov and Van Voorhis 2010 eq. 25-26
    # https://doi.org/10.1063/1.3521275

    #output
    exc=cupy.zeros((rho[0,:].size,3))

    #outer grid needs threshing
    threshind=rho[0,:]>=NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
    coords=coords[threshind]
    R=rho[0,:][threshind]
    Gx=rho[1,:][threshind]
    Gy=rho[2,:][threshind]
    Gz=rho[3,:][threshind]
    G=Gx**2.+Gy**2.+Gz**2.

    #inner grid needs threshing
    innerthreshind=vvrho[0,:]>=NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
    vvcoords=vvcoords[innerthreshind]
    vvweight=vvweight[innerthreshind]
    Rp=vvrho[0,:][innerthreshind]
    RpW=Rp*vvweight
    Gxp=vvrho[1,:][innerthreshind]
    Gyp=vvrho[2,:][innerthreshind]
    Gzp=vvrho[3,:][innerthreshind]
    Gp=Gxp**2.+Gyp**2.+Gzp**2.

    #constants and parameters
    Pi=numpy.pi
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
    K=Kvv*(R**(1./6.))

    vvcoords = cupy.asarray(vvcoords, order='C')
    coords = cupy.asarray(coords, order='C')
    F = cupy.empty((R.shape[0], 3), order='C')
    stream = cupy.cuda.get_current_stream()
    libgdft.VXC_vv10nlc_grad(ctypes.cast(stream.ptr, ctypes.c_void_p),
                             ctypes.cast(F.data.ptr, ctypes.c_void_p),
                             ctypes.cast(vvcoords.data.ptr, ctypes.c_void_p),
                             ctypes.cast(coords.data.ptr, ctypes.c_void_p),
                             ctypes.cast(W0p.data.ptr, ctypes.c_void_p),
                             ctypes.cast(W0.data.ptr, ctypes.c_void_p),
                             ctypes.cast(K.data.ptr, ctypes.c_void_p),
                             ctypes.cast(Kp.data.ptr, ctypes.c_void_p),
                             ctypes.cast(RpW.data.ptr, ctypes.c_void_p),
                             ctypes.c_int(vvcoords.shape[0]),
                             ctypes.c_int(coords.shape[0]))
    #exc is multiplied by Rho later
    exc[threshind] = F
    return exc, Beta

def get_nlc_exc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                              max_memory=2000, verbose=None):
    '''Full NLC functional response including the response of the grids'''
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    dms = cupy.asarray(dms)
    assert dms.ndim == 2
    dms = opt.sort_orbitals(dms, axis=[0,1])

    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]
    ao_deriv = 2

    excsum = cupy.zeros((mol.natm, 3))
    vmat = cupy.zeros((3,nao,nao))

    vvrho = []
    vvcoords = []
    vvweights = []
    for atm_id, (coords, weight) in enumerate(grids_noresponse_cc(grids)):
        ao = ni.eval_ao(_sorted_mol, coords, ao_deriv, gdftopt=opt, transpose=False)
        rho = numint.eval_rho(_sorted_mol, ao[:4], dms, xctype=xctype, hermi=1, with_lapl=False)
        vvrho.append(rho)
        vvcoords.append(coords)
        vvweights.append(weight)
    vvcoords_flat = cupy.vstack(vvcoords)
    vvweights_flat = cupy.concatenate(vvweights)
    vvrho_flat = cupy.hstack(vvrho)

    mem_avail = get_avail_mem()
    comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
    block_size = int((mem_avail*.4/8/(comp+1)/nao - 3*nao*2)/ ALIGNED) * ALIGNED
    block_size = min(block_size, MIN_BLK_SIZE)
    log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)

    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        ngrids = weight.size
        for p0, p1 in lib.prange(0,ngrids,block_size):
            ao = numint.eval_ao(_sorted_mol, coords[p0:p1, :], ao_deriv, gdftopt=opt, transpose=False)

            rho = numint.eval_rho(_sorted_mol, ao[:4], dms, xctype=xctype, hermi=1, with_lapl=False)

            exc, vxc = numint._vv10nlc(rho, coords[p0:p1, :], vvrho_flat, vvweights_flat,
                                        vvcoords_flat, nlc_pars)
            vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

            wv = weight[p0:p1] * vv_vxc
            wv[0] *= .5
            vtmp = _gga_grad_sum_(ao, wv)
            vmat += vtmp

            vvrho_sub = cupy.hstack(
                [r for i, r in enumerate(vvrho) if i != atm_id])
            vvcoords_sub = cupy.vstack(
                [r for i, r in enumerate(vvcoords) if i != atm_id])
            vvweights_sub = cupy.concatenate(
                [r for i, r in enumerate(vvweights) if i != atm_id])
            egrad, Beta = _vv10nlc_grad(rho, coords[p0:p1, :], vvrho_sub,
                                        vvweights_sub, vvcoords_sub, nlc_pars)

            # account for factor of 2 in double integration
            exc -= 0.5 * Beta
            # response of weights
            excsum += 2 * cupy.einsum('r,nxr->nx', exc * rho[0], weight1[:,:,p0:p1])
            # response of grids coordinates
            excsum[atm_id] += 2 * cupy.einsum('xij,ji->x', vtmp, dms)
            excsum[atm_id] += cupy.einsum('r,rx->x', rho[0]*weight[p0:p1], egrad)

    # - sign because nabla_X = -nabla_x
    exc1 = -.5 * rhf_grad.contract_h1e_dm(opt._sorted_mol, vmat, dms, hermi=1)
    log.timer_debug1('grad nlc vxc full response', *t0)
    return excsum.get(), exc1

# JCP 98, 5612 (1993); DOI:10.1063/1.464906
def grids_response_cc(grids):
    # Notice: the returned grid order could be different from pyscf.grad.rks.grids_response_cc()!
    assert grids.becke_scheme == gen_grid.original_becke
    mol = grids.mol

    grid_to_atom_index_map = grids.atm_idx
    atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(mol.natm)]
    grid_to_atom_index_map = None

    from gpu4pyscf.hessian.rks import get_dweight_dA # Avoid circular dependency

    for i_atom in range(mol.natm):
        i_g = atom_to_grid_index_map[i_atom]
        fake_grids = type('FakeGrid', (object,), {})()
        fake_grids.coords = grids.coords[i_g, :]
        fake_grids.weights = grids.weights[i_g]
        fake_grids.quadrature_weights = grids.quadrature_weights[i_g]
        fake_grids.atm_idx = cupy.zeros(len(i_g), dtype = cupy.int32) + i_atom
        fake_grids.atomic_radii = grids.atomic_radii
        dw_dA_i = get_dweight_dA(mol, fake_grids)
        yield fake_grids.coords, fake_grids.weights, dw_dA_i

def grids_noresponse_cc(grids):
    # same as above but without the response, for nlc grids response routine
    # Similarly, the returned grid order could be different from pyscf.grad.rks.grids_noresponse_cc()!
    assert grids.becke_scheme == gen_grid.original_becke
    mol = grids.mol

    grid_to_atom_index_map = grids.atm_idx
    atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(mol.natm)]
    grid_to_atom_index_map = None

    for i_atom in range(mol.natm):
        i_g = atom_to_grid_index_map[i_atom]
        yield grids.coords[i_g, :], grids.weights[i_g]

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

Grad = Gradients
