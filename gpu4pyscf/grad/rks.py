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
    reduce_to_device, take_last2d, ndarray, batched_vec_norm2)
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
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=0)[1][0]
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
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=0, work=aow_buf)[1]
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
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=0, work=aow_buf)[1]
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
    xctype = "GGA"
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

    ao_deriv = 1
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

    vxc = numint._vv10nlc(rho, grids.coords, grids.weights, nlc_pars)[1]
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    ao_deriv = 2
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
    assert ao.shape[0] >= 10
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
    t0 = log.init_timer()
    ni = numint.NumInt() # Don't mess up with the old numint object
    xctype = ni._xc_type(xc_code)

    grids = grids.copy()
    grids.build(sort_grids_of_each_atom = True)
    ngrids = grids.coords.shape[0]

    ni.gdftopt = None
    ni.build(mol, grids.coords)
    opt = ni.gdftopt

    natm = mol.natm
    mol = None
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    dms = cupy.asarray(dms)
    assert dms.ndim == 2
    dms = opt.sort_orbitals(dms, axis=[0,1])

    de_grid_response_rho = cupy.zeros((natm, 3))
    dvmat_orbital_response = cupy.zeros((3, nao, nao))
    dm_mask_buf = cupy.empty(nao*nao)

    if xctype == 'LDA':
        ao_deriv = 0
        ncomp = 1
    elif xctype == 'GGA':
        ao_deriv = 1
        ncomp = 4
    elif xctype == 'MGGA':
        ao_deriv = 1
        ncomp = 5
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    rho = cupy.empty([ncomp, ngrids])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = ao_deriv, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        dms_masked = take_last2d(dms, idx, out=dm_mask_buf)
        rho[:, g0:g1] = numint.eval_rho(_sorted_mol, ao, dms_masked, xctype = xctype, hermi = 1)
    assert g1 == ngrids

    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=0)[:2]
    wv = grids.weights * vxc
    nonzero_weight_mask = cupy.abs(grids.weights) > 1e-14

    from gpu4pyscf.hessian.rks import get_dweight_dA

    de_grid_response_weight = cupy.zeros((natm, 3))
    dweightdA_right = rho[0] * exc
    del rho

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.1) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((2*3) * natm) * 8
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for XC energy first derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {nao}, natm = {natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ### Don't split the batch too small for get_dweight_dA()
    # ngrids_per_batch = min(ngrids_per_batch, MIN_BLK_SIZE)

    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        dweight_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
        de_grid_response_weight += cupy.einsum("Adg->Ad", dweight_dA * dweightdA_right[g0:g1])
    del dweight_dA
    del dweightdA_right
    del exc

    g0 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv + 1, strict_grid_order = True):
        g1 = g0 + weight.shape[0]

        ao = ao[:, :, nonzero_weight_mask[g0:g1]]

        if ao.size == 0:
            g0 = g1
            continue

        dms_masked = take_last2d(dms, idx, out=dm_mask_buf)

        i_atom = int(grids.atm_idx[g0])
        assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

        if xctype == 'LDA':
            split_wv = cupy.ascontiguousarray(wv[:, g0:g1][:, nonzero_weight_mask[g0:g1]])

            aow = numint._scale_ao(ao[0], split_wv[0])
            vtmp = _d1_dot_(ao[1:4], aow.T)

            dvmat_orbital_response[:, idx[:,None], idx] += vtmp
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dms_masked) * 2

        elif xctype == 'GGA':
            split_wv = cupy.ascontiguousarray(wv[:, g0:g1][:, nonzero_weight_mask[g0:g1]])
            split_wv[0] *= .5

            vtmp = _gga_grad_sum_(ao, split_wv[:4])

            dvmat_orbital_response[:, idx[:,None], idx] += vtmp
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dms_masked) * 2

        elif xctype == 'NLC':
            raise ValueError("You see a bug, please report to the developer team.")

        elif xctype == 'MGGA':
            split_wv = cupy.ascontiguousarray(wv[:, g0:g1][:, nonzero_weight_mask[g0:g1]])
            split_wv[0] *= .5
            split_wv[4] *= .5 # for the factor 1/2 in tau

            vtmp = _gga_grad_sum_(ao, split_wv[:4])
            _tau_grad_dot_(ao, split_wv[4], accumulate=True, out=vtmp)

            dvmat_orbital_response[:, idx[:,None], idx] += vtmp
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dms_masked) * 2

        else:
            raise NotImplementedError(f"Unrecognized xctype = {xctype}")

        g0 = g1
    assert g1 == ngrids

    excsum = de_grid_response_weight + de_grid_response_rho
    excsum = excsum.get()
    # - sign because nabla_X = -nabla_x
    excsum -= rhf_grad.contract_h1e_dm(opt._sorted_mol, dvmat_orbital_response, dms, hermi=1)

    log.timer_debug1('rks grad vxc full response', *t0)
    return excsum, 0

def get_nlc_exc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                              max_memory=2000, verbose=None):
    '''Full NLC functional response including the response of the grids'''

    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    grids = grids.copy()
    grids.build(sort_grids_of_each_atom = True)

    ni = numint.NumInt() # Don't mess up with the old numint object
    ni.gdftopt = None
    ni.build(mol, grids.coords)
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    natm = _sorted_mol.natm
    mol = None
    dms = cupy.asarray(dms).reshape(-1,nao,nao)
    nset = len(dms)
    assert nset == 1 or nset == 2
    if nset == 1:
        dms = dms[0]
    else:
        dms = dms[0] + dms[1]
    dms_sorted = opt.sort_orbitals(dms, axis=[0,1])
    dm_mask_buf = cupy.empty(nao * nao)
    dms = None

    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        dms_masked = take_last2d(dms_sorted, idx, out = dm_mask_buf)
        rho_drho[:, g0:g1] = numint.eval_rho(_sorted_mol, ao, dms_masked, xctype = "NLC", hermi = 1)
    assert g1 == ngrids_full

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )

    rho_i = rho_i[rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    nabla_rho_i = cupy.ascontiguousarray(rho_drho[1:4, rho_nonzero_mask])
    gamma_i = batched_vec_norm2(nabla_rho_i.T)

    omega_i         = cupy.empty(ngrids)
    domega_drho_i   = cupy.empty(ngrids)
    domega_dgamma_i = cupy.empty(ngrids)
    stream = cupy.cuda.get_current_stream()
    err = libgdft.VXC_vv10nlc_fock_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(gamma_i.data.ptr, ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids),
    )
    if err != 0:
        raise RuntimeError('CUDA Error in vv10 gradient (grid response) kernel')
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i = (kappa_prefactor * (1.0/6.0)) * rho_i**(-5.0/6.0)

    rho_weight_i = rho_i * grids_weights

    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)
    err = libgdft.VXC_vv10nlc_fock_eval_UWE(
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
    if err != 0:
        raise RuntimeError('CUDA Error in vv10 gradient (grid response) kernel')

    fw_rho_i = (beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)) * grids_weights
    fw_gamma_i = rho_i * domega_dgamma_i * W_i * grids_weights
    del dkappa_drho_i, domega_drho_i, domega_dgamma_i
    del U_i, W_i

    fw_gamma_vxc_form = 2 * nabla_rho_i * fw_gamma_i
    del fw_gamma_i, nabla_rho_i

    dvmat_orbital_response = cupy.zeros((3, nao, nao))
    de_grid_response_rho = cupy.zeros((natm, 3))

    g0_full = 0
    g0_nonzero = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 2, strict_grid_order = True):
        g1_full = g0_full + weight.shape[0]

        ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

        if ao.size == 0:
            g0_full = g1_full
            continue

        g1_nonzero = g0_nonzero + ao.shape[-1]

        i_atom_of_grids = int(grids.atm_idx[g0_full])
        assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

        wv = cupy.vstack([fw_rho_i[g0_nonzero : g1_nonzero], fw_gamma_vxc_form[:, g0_nonzero : g1_nonzero]])
        wv[0] *= .5
        vtmp = _gga_grad_sum_(ao, wv)
        dvmat_orbital_response[numpy.ix_(range(3), idx, idx)] += vtmp
        dms_masked = take_last2d(dms_sorted, idx, out = dm_mask_buf)
        de_grid_response_rho[i_atom_of_grids] += cupy.einsum('xij,ji->x', vtmp, dms_masked) * 2

        del wv, vtmp

        g0_nonzero = g1_nonzero
        g0_full = g1_full
    assert g1_full == ngrids_full
    assert g1_nonzero == ngrids

    from gpu4pyscf.hessian.rks import get_dweight_dA

    de_grid_response_weight = cupy.zeros((natm, 3))
    dweightdA_right = rho_i * (beta + E_i)

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((2*3) * natm) * 8
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC energy first derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {nao}, natm = {natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ### Don't split the batch too small, it'll damage the performance of VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal kernel
    # ngrids_per_batch = min(ngrids_per_batch, MIN_BLK_SIZE)

    g0_nonzero = 0
    for g0_full in range(0, ngrids_full, ngrids_per_batch):
        g1_full = min(g0_full + ngrids_per_batch, ngrids_full)

        dweight_dA = get_dweight_dA(_sorted_mol, grids, (g0_full, g1_full))
        dweight_dA = dweight_dA[:, :, rho_nonzero_mask[g0_full : g1_full]]

        g1_nonzero = g0_nonzero + dweight_dA.shape[2]
        de_grid_response_weight += cupy.einsum("Adg->Ad", dweight_dA * dweightdA_right[g0_nonzero : g1_nonzero])

        g0_nonzero = g1_nonzero
        del dweight_dA
    del dweightdA_right
    assert g1_nonzero == ngrids

    grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]
    grid_offsets_of_atom = cupy.r_[0, cupy.flatnonzero(cupy.diff(grid_to_atom_index_map)) + 1]
    if grid_to_atom_index_map[-1] < 0:
        pass # There's padded grids whose index < 0, and the first index of padded grids is the number of valid grids
    else:
        grid_offsets_of_atom = cupy.append(grid_offsets_of_atom, grid_to_atom_index_map.shape[0])
    grid_offsets_of_atom = cupy.asarray(grid_offsets_of_atom, dtype = cupy.int32)

    assert grid_offsets_of_atom.shape == (natm + 1,)
    for i_atom in range(natm):
        assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[i_atom] : grid_offsets_of_atom[i_atom + 1]] == i_atom)
    assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[natm] : ] < 0)

    de_grid_response_phi = cupy.zeros((natm, 3))

    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)

        E_Bgr_i = cupy.empty([natm, 3, g1-g0], order = "C")
        err = libgdft.VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_offsets_of_atom.data.ptr, ctypes.c_void_p),
            ctypes.c_int(natm),
            ctypes.c_int(g0),
            ctypes.c_int(g1-g0),
        )
        if err != 0:
            raise RuntimeError('CUDA Error in vv10 gradient (grid response) kernel')

        for i_atom in range(natm):
            range_0, range_1 = grid_offsets_of_atom[i_atom] - g0, grid_offsets_of_atom[i_atom + 1] - g0
            range_0 = max(range_0, 0)
            range_1 = min(range_1, g1)
            if range_0 >= g1 or range_1 < 0:
                continue

            E_Bgr_i[i_atom, :, range_0:range_1] = \
                -cupy.sum(E_Bgr_i[:, :, range_0:range_1], axis = 0)

        de_grid_response_phi += cupy.einsum("Adg->Ad", E_Bgr_i * rho_weight_i[g0:g1])

    del omega_i, kappa_i
    del rho_weight_i

    exc1 = de_grid_response_rho + de_grid_response_weight + de_grid_response_phi
    exc1 = exc1.get()
    exc1 += -rhf_grad.contract_h1e_dm(_sorted_mol, dvmat_orbital_response, dms_sorted, hermi=1)

    log.timer_debug1('grad nlc vxc full response', *t0)

    return exc1, 0

def grids_response_cc(grids):
    raise NotImplementedError("grids_response_cc() in GPU4PySCF is not used or tested anymore")

def grids_noresponse_cc(grids):
    raise NotImplementedError("grids_noresponse_cc() in GPU4PySCF is not used or tested anymore")

class Gradients(rhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    grid_response = rks_grad.Gradients.grid_response

    _keys = rks_grad.Gradients._keys

    def __init__ (self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    energy_ee = energy_ee

Grad = Gradients
