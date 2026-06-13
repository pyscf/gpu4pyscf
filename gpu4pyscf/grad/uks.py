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
# Modified by Xiaojie Wu <wxj6000@gmail.com>, Zhichen Pu <hoshishin@163.com>

'''Non-relativistic UKS analytical nuclear gradients'''
from concurrent.futures import ThreadPoolExecutor
import ctypes
import cupy
from pyscf import lib
from pyscf.grad import uks as uks_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import eval_rho2
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, add_sparse, tag_array, reduce_to_device,
    take_last2d, ndarray)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf import __config__

MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 128*128)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)

libgdft = rks_grad.libgdft
libgdft.GDFT_make_dR_dao_w.restype = ctypes.c_int

def energy_ee(ks_grad, mol=None, dm=None, verbose=None):
    '''
    First-order derivatives of the two-electron energy contributions per atom

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
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
        #logger.debug1(ks_grad, 'grids response %s', exc)
        exc1 *= 2
        exc1 += exc
    else:
        exc, exc1 = get_exc(ni, mol, grids, mf.xc, dm,
                            verbose=ks_grad.verbose)
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

def _get_exc_task(ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                  verbose=None, with_lapl=False, grid_range=(), device_id=0):
    ''' Calculate the gradient of vxc on given device
    '''
    with cupy.cuda.Device(device_id):
        if dms is not None: dms = cupy.asarray(dms)
        if mo_coeff is not None: mo_coeff = cupy.asarray(mo_coeff)
        if mo_occ is not None: mo_occ = cupy.asarray(mo_occ)

        log = logger.new_logger(mol)
        t0 = log.init_timer()
        xctype = ni._xc_type(xc_code)
        nao = mol.nao
        opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol

        nocc_a = cupy.count_nonzero(mo_occ[0]>0)
        nocc_b = cupy.count_nonzero(mo_occ[1]>0)
        nocc = max(nocc_a, nocc_b)

        ngrids_glob = grids.coords.shape[0]
        grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
        ngrids_local = grid_end - grid_start
        log.debug(f"{ngrids_local} grids on Device {device_id}")

        exc1 = cupy.zeros((nao, 3))

        if xctype == 'LDA':
            ncomp = 1
        elif xctype == 'GGA':
            ncomp = 4
        else:
            ncomp = 5
        rho_buf = cupy.empty(2*ncomp*MIN_BLK_SIZE)
        mo_buf = cupy.empty_like(mo_coeff[0])
        vtmp_buf = cupy.empty((3, nao, nao))

        dm_mask_buf = cupy.empty(nao*nao)

        if xctype == 'LDA':
            ao_deriv = 1
            aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao, 1*nocc))
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                blk_size = len(weight)
                nao_sub = len(idx)
                rho = ndarray((2, ncomp, blk_size), buffer=rho_buf)
                mo_coeff_mask = cupy.take(mo_coeff[0], idx, axis=0, out=mo_buf[:nao_sub])
                eval_rho2(_sorted_mol, ao_mask[0], mo_coeff_mask, mo_occ[0], None, xctype, buf=aow_buf, out=rho[0])
                mo_coeff_mask = cupy.take(mo_coeff[1], idx, axis=0, out=mo_buf[:nao_sub])
                eval_rho2(_sorted_mol, ao_mask[0], mo_coeff_mask, mo_occ[1], None, xctype, buf=aow_buf, out=rho[1])

                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=1)[1][:,0]
                wv = cupy.multiply(weight, vxc, out=vxc)
                aow = numint._scale_ao(ao_mask[0], wv[0], out=aow_buf)
                vtmp = rks_grad._d1_dot_(ao_mask[1:4], aow.T, out=vtmp_buf)
                dm_mask = take_last2d(dms[0], idx, out=dm_mask_buf)
                exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
                aow = numint._scale_ao(ao_mask[0], wv[1], out=aow)
                vtmp = rks_grad._d1_dot_(ao_mask[1:4], aow.T, out=vtmp_buf)
                dm_mask = take_last2d(dms[1], idx, out=dm_mask)
                exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
        elif xctype == 'GGA':
            ao_deriv = 2
            aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao * 3, 2*nocc))
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                blk_size = len(weight)
                nao_sub = len(idx)
                rho = ndarray((2, ncomp, blk_size), buffer=rho_buf)
                mo_coeff_mask = cupy.take(mo_coeff[0], idx, axis=0, out=mo_buf[:nao_sub])
                eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ[0], None, xctype, buf=aow_buf, out=rho[0])
                mo_coeff_mask = cupy.take(mo_coeff[1], idx, axis=0, out=mo_buf[:nao_sub])
                eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ[1], None, xctype, buf=aow_buf, out=rho[1])

                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=1)[1]
                wv = cupy.multiply(weight, vxc, out=vxc)
                wv[:,0] *= .5
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[0], buf=aow_buf, out=vtmp_buf)
                #add_sparse(vmat[0], vtmp, idx)
                dm_mask = take_last2d(dms[0], idx, out=dm_mask_buf)
                exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[1], buf=aow_buf, out=vtmp_buf)
                #add_sparse(vmat[1], vtmp, idx)
                dm_mask = take_last2d(dms[1], idx, out=dm_mask)
                exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 2
            aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao * 3, 2*nocc))
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                blk_size = len(weight)
                nao_sub = len(idx)
                rho = ndarray((2, ncomp, blk_size), buffer=rho_buf)
                mo_coeff_mask = cupy.take(mo_coeff[0], idx, axis=0, out=mo_buf[:nao_sub])
                eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ[0], None, xctype, buf=aow_buf, out=rho[0])
                mo_coeff_mask = cupy.take(mo_coeff[1], idx, axis=0, out=mo_buf[:nao_sub])
                eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask, mo_occ[1], None, xctype, buf=aow_buf, out=rho[1])

                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=1)[1]
                wv = cupy.multiply(weight, vxc, out=vxc)
                wv[:,0] *= .5
                wv[:,4] *= .5  # for the factor 1/2 in tau
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[0], buf=aow_buf, out=vtmp_buf)
                vtmp = rks_grad._tau_grad_dot_(ao_mask, wv[0,4], accumulate=True, buf=aow_buf, out=vtmp)
                #add_sparse(vmat[0], vtmp, idx)
                dm_mask = take_last2d(dms[0], idx, out=dm_mask_buf)
                exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[1], buf=aow_buf, out=vtmp_buf)
                vtmp = rks_grad._tau_grad_dot_(ao_mask, wv[1,4], accumulate=True, buf=aow_buf, out=vtmp)
                #add_sparse(vmat[1], vtmp, idx)
                dm_mask = take_last2d(dms[1], idx, out=dm_mask)
                exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T
        log.timer_debug1('uks gradient of vxc', *t0)
    return exc1

def get_exc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_occ = cupy.asarray(dms.mo_occ)
    mo_coeff = cupy.asarray(dms.mo_coeff)
    dms = cupy.asarray(dms)
    dms = opt.sort_orbitals(dms, axis=[1,2])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_exc_task,
                ni, mol, grids, xc_code, dms, mo_coeff, mo_occ,
                verbose=verbose, device_id=device_id)
            futures.append(future)
    vmat_dist = [future.result() for future in futures]
    vmat = reduce_to_device(vmat_dist)
    exc = None
    # - sign because nabla_X = -nabla_x
    exc1 = -rks_grad._reduce_to_atom(opt._sorted_mol, vmat)
    return exc, exc1

def get_exc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    ni = numint.NumInt() # Don't mess up with the old ni
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
    assert dms.ndim == 3 and dms.shape[0] == 2
    dms = opt.sort_orbitals(dms.reshape(-1,nao,nao), axis=[1,2])

    de_grid_response_rho = cupy.zeros((natm, 3))
    dvmat_orbital_response = cupy.zeros((2, 3, nao, nao))
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

    rho = cupy.empty([2, ncomp, ngrids])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = ao_deriv, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        dma_masked = take_last2d(dms[0], idx, out=dm_mask_buf)
        rho[0, :, g0:g1] = numint.eval_rho(_sorted_mol, ao, dma_masked, xctype = xctype, hermi = 1)
        dmb_masked = take_last2d(dms[1], idx, out=dm_mask_buf)
        rho[1, :, g0:g1] = numint.eval_rho(_sorted_mol, ao, dmb_masked, xctype = xctype, hermi = 1)
    assert g1 == ngrids

    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype, spin=1)[:2]
    wv = grids.weights * vxc
    nonzero_weight_mask = cupy.abs(grids.weights) > 1e-14

    from gpu4pyscf.hessian.rks import get_dweight_dA

    de_grid_response_weight = cupy.zeros((natm, 3))
    dweightdA_right = (rho[0,0] + rho[1,0]) * exc
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

        i_atom = int(grids.atm_idx[g0])
        assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

        if xctype == 'LDA':
            split_wv = cupy.ascontiguousarray(wv[:, :, g0:g1][:, :, nonzero_weight_mask[g0:g1]])

            aow = numint._scale_ao(ao[0], split_wv[0, 0])
            vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
            dvmat_orbital_response[0][:, idx[:,None], idx] += vtmp
            dma_masked = take_last2d(dms[0], idx, out=dm_mask_buf)
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dma_masked) * 2

            aow = numint._scale_ao(ao[0], split_wv[1, 0])
            vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
            dvmat_orbital_response[1][:, idx[:,None], idx] += vtmp
            dmb_masked = take_last2d(dms[1], idx, out=dm_mask_buf)
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dmb_masked) * 2

        elif xctype == 'GGA':
            split_wv = cupy.ascontiguousarray(wv[:, :, g0:g1][:, :, nonzero_weight_mask[g0:g1]])
            split_wv[:, 0] *= .5

            vtmp = rks_grad._gga_grad_sum_(ao, split_wv[0])
            dvmat_orbital_response[0][:, idx[:,None], idx] += vtmp
            dma_masked = take_last2d(dms[0], idx, out=dm_mask_buf)
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dma_masked) * 2

            vtmp = rks_grad._gga_grad_sum_(ao, split_wv[1])
            dvmat_orbital_response[1][:, idx[:,None], idx] += vtmp
            dmb_masked = take_last2d(dms[1], idx, out=dm_mask_buf)
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dmb_masked) * 2

        elif xctype == 'NLC':
            raise ValueError("You see a bug, please report to the developer team.")

        elif xctype == 'MGGA':
            split_wv = cupy.ascontiguousarray(wv[:, :, g0:g1][:, :, nonzero_weight_mask[g0:g1]])
            split_wv[:, 0] *= .5
            split_wv[:, 4] *= .5 # for the factor 1/2 in tau

            vtmp = rks_grad._gga_grad_sum_(ao, split_wv[0, :4])
            rks_grad._tau_grad_dot_(ao, split_wv[0, 4], accumulate=True, out=vtmp)
            dvmat_orbital_response[0][:, idx[:,None], idx] += vtmp
            dma_masked = take_last2d(dms[0], idx, out=dm_mask_buf)
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dma_masked) * 2

            vtmp = rks_grad._gga_grad_sum_(ao, split_wv[1, :4])
            rks_grad._tau_grad_dot_(ao, split_wv[1, 4], accumulate=True, out=vtmp)
            dvmat_orbital_response[1][:, idx[:,None], idx] += vtmp
            dmb_masked = take_last2d(dms[1], idx, out=dm_mask_buf)
            de_grid_response_rho[i_atom] += cupy.einsum('xij,ji->x', vtmp, dmb_masked) * 2

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

_get_denlc = rks_grad._get_denlc
get_nlc_exc = rks_grad.get_nlc_exc
get_nlc_exc_full_response = rks_grad.get_nlc_exc_full_response

class Gradients(uhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    grid_response = uks_grad.Gradients.grid_response

    _keys = uks_grad.Gradients._keys

    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    energy_ee = energy_ee

Grad = Gradients
