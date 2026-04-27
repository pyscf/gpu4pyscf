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
    reduce_to_device, take_last2d, ndarray, batched_vec3_norm2)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
from gpu4pyscf.gto.mole import groupby, ATOM_OF

from pyscf import __config__
MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 4096)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)

libgdft = numint.libgdft
libgdft.GDFT_make_dR_dao_w.restype = ctypes.c_int

def energy_ee(ks_grad, mol=None, dm=None, verbose=None):
    '''
    Computes the first-order derivatives of the two-electron energy
    contributions per atom

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
        exc1 *= 2
        exc1 += exc
    else:
        exc, exc1 = get_exc(ni, mol, grids, mf.xc, dm, verbose=log)
        exc1 *= 2
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if mf.do_nlc():
        enlc1_per_atom, enlc1_grid = _get_denlc(ks_grad, mol, dm)
        exc1 += enlc1_per_atom * 2
        if ks_grad.grid_response:
            exc1 += enlc1_grid

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
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
    exc1 += ks_grad.jk_energy_per_atom(dm, j_factor, k_factor, verbose=log)

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
        exc1 += ks_grad.jk_energy_per_atom(
            dm, j_factor, k_factor, omega=omega, verbose=log)
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
    nset = len(dms)
    assert nset == 1 or nset == 2
    if nset == 1:
        dm = dms[0]
        dms = None
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    else:
        dm = dms[0] + dms[1]
        dms = None
        mo_coeff_0 = opt.sort_orbitals(mo_coeff[0], axis=[0])
        mo_coeff_1 = opt.sort_orbitals(mo_coeff[1], axis=[0])

    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    ao_deriv = 2
    vvrho = []
    for ao_mask, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=max_memory):
        if nset == 1:
            mo_coeff_mask = mo_coeff[mask]
            rho = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ, None, xctype, with_lapl=False)
            vvrho.append(rho)
        else:
            mo_coeff_mask_0 = mo_coeff_0[mask]
            mo_coeff_mask_1 = mo_coeff_1[mask]
            rhoa = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask_0, mo_occ[0], None, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask_1, mo_occ[1], None, xctype)
            vvrho.append(rhoa + rhob)
    rho = cupy.hstack(vvrho)
    vvrho = None

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

def get_nlc_exc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                              max_memory=2000, verbose=None):
    '''Full NLC functional response including the response of the grids'''

    import time
    cupy.cuda.runtime.deviceSynchronize()
    time_0 = time.time()

    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    dms_before_sort = dms.copy()
    if dms_before_sort.ndim > 2:
        dms_before_sort = dms_before_sort[0] + dms_before_sort[1]

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    dms = cupy.asarray(dms).reshape(-1,nao,nao)
    dms = opt.sort_orbitals(dms, axis=[1,2])
    nset = len(dms)
    assert nset == 1 or nset == 2
    if nset == 1:
        dms = dms[0]
    else:
        dms = dms[0] + dms[1]

    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_preparation = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(_sorted_mol, grids, deriv = 1):
        g0, g1 = g1, g1 + split_weights.size
        dms_masked = dms[ao_mask_index[:,None], ao_mask_index]
        rho_drho[:, g0:g1] = numint.eval_rho(_sorted_mol, split_ao, dms_masked, xctype = "NLC", hermi = 1)

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = cupy.ascontiguousarray(rho_drho[1:4, rho_nonzero_mask])
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = batched_vec3_norm2(nabla_rho_i)

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_rho_gamma = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    omega_i         = cupy.empty(ngrids)
    domega_drho_i   = cupy.empty(ngrids)
    domega_dgamma_i = cupy.empty(ngrids)
    stream = cupy.cuda.get_current_stream()
    libgdft.VXC_vv10nlc_fock_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(gamma_i.data.ptr, ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids),
    )
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_domega = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    rho_weight_i = rho_i * grids_weights
    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_fock_eval_UWE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
    )
    del rho_weight_i

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_UWE = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    fw_rho_i = (beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)) * grids_weights
    fw_gamma_i = rho_i * domega_dgamma_i * W_i * grids_weights

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_fw = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    from gpu4pyscf.hessian.rks import get_dweight_dA, get_d2mu_dr2, get_drhodA_dgammadA_orbital_response, get_drhodA_dgammadA_grid_response

    dweight_dA = get_dweight_dA(mol, grids)
    dweight_dA = dweight_dA[:, :, rho_nonzero_mask]

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_dweight = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    aoslices = mol.aoslice_by_atom()

    grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]
    atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(mol.natm)]

    drho_dA_full_response   = cupy.empty([mol.natm, 3, ngrids], order = "C")
    dgamma_dA_full_response = cupy.empty([mol.natm, 3, ngrids], order = "C")

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((10 + 1*4 + 3*4 + 9) * mol.nao + (3*4) * mol.natm) * 8
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC energy second derivative, available gpu memory = {get_avail_mem()}"
                            f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, MIN_BLK_SIZE)

    from gpu4pyscf.dft.numint import NumInt
    empty_numint = NumInt()
    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        split_grids_coords = grids_coords[g0:g1, :]
        split_ao = empty_numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

        mu = split_ao[0, :, :]
        dmu_dr = split_ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(split_ao)
        split_drho_dr = nabla_rho_i[:, g0:g1]
        split_grid_to_atom_index_map = grid_to_atom_index_map[g0:g1]
        split_atom_to_grid_index_map = [cupy.where(split_grid_to_atom_index_map == i_atom)[0] for i_atom in range(mol.natm)]

        split_drho_dA_orbital_response, split_dgamma_dA_orbital_response = \
            get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dms_before_sort, aoslices)
        split_drho_dA_grid_response,    split_dgamma_dA_grid_response = \
            get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dms_before_sort, split_atom_to_grid_index_map)

        drho_dA_full_response  [:, :, g0:g1] =   split_drho_dA_orbital_response +   split_drho_dA_grid_response
        dgamma_dA_full_response[:, :, g0:g1] = split_dgamma_dA_orbital_response + split_dgamma_dA_grid_response
        split_ao = None
        mu = None
        dmu_dr = None
        d2mu_dr2 = None
        split_drho_dA_orbital_response   = None
        split_dgamma_dA_orbital_response = None
        split_drho_dA_grid_response   = None
        split_dgamma_dA_grid_response = None

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_drho = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    rho_weight_i = rho_i * grids_weights
    E_Bgr_i = cupy.empty([mol.natm, 3, ngrids], order = "C")
    libgdft.VXC_vv10nlc_grad_eval_E_grid_response(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(E_Bgr_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
    )
    del rho_weight_i

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_dphi = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    exc1  = cupy.einsum("Adg->Ad", drho_dA_full_response * fw_rho_i + dgamma_dA_full_response * fw_gamma_i)
    exc1 += cupy.einsum("Adg->Ad", dweight_dA * rho_i * (beta + E_i))
    exc1 += cupy.einsum("Adg->Ad", E_Bgr_i * rho_i * grids_weights)

    exc1 = exc1.get()

    log.timer_debug1('grad nlc vxc full response', *t0)

    cupy.cuda.runtime.deviceSynchronize()
    time_1 = time.time()
    print(f"time_postprocess = {time_1 - time_0}")
    print("\n")
    time_0 = time_1

    return exc1, 0

# JCP 98, 5612 (1993); DOI:10.1063/1.464906
def grids_response_cc(grids):
    # Notice: the returned grid order could be different from pyscf.grad.rks.grids_response_cc()!
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
        fake_grids.radii_adjust = grids.radii_adjust
        fake_grids.becke_scheme = grids.becke_scheme
        dw_dA_i = get_dweight_dA(mol, fake_grids)
        yield fake_grids.coords, fake_grids.weights, dw_dA_i

def grids_noresponse_cc(grids):
    # same as above but without the response, for nlc grids response routine
    # Similarly, the returned grid order could be different from pyscf.grad.rks.grids_noresponse_cc()!
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

    energy_ee = energy_ee

Grad = Gradients
