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
from gpu4pyscf.dft import numint, xc_deriv
from gpu4pyscf.dft.numint import eval_rho2
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, add_sparse, tag_array, reduce_to_device)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, _num_devices
from gpu4pyscf import __config__

MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 128*128)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)

libgdft = rks_grad.libgdft
libgdft.GDFT_make_dR_dao_w.restype = ctypes.c_int

def get_veff(ks_grad, mol=None, dm=None, verbose=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
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
        exc, vxc_tmp = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            raise NotImplementedError
    else:
        exc, vxc_tmp = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm, mf.mo_coeff, mf.mo_occ,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc_tmp[0] += vnlc
            vxc_tmp[1] += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    mo_coeff_alpha = mf.mo_coeff[0]
    mo_coeff_beta = mf.mo_coeff[1]
    occ_coeff0 = cupy.asarray(mo_coeff_alpha[:, mf.mo_occ[0]>0.5], order='C')
    occ_coeff1 = cupy.asarray(mo_coeff_beta[:, mf.mo_occ[1]>0.5], order='C')
    tmp = contract('nij,jk->nik', vxc_tmp[0], occ_coeff0)
    exc1_per_atom = contract('nik,ik->ni', tmp, occ_coeff0)
    tmp = contract('nij,jk->nik', vxc_tmp[1], occ_coeff1)
    exc1_per_atom+= contract('nik,ik->ni', tmp, occ_coeff1)

    aoslices = mol.aoslice_by_atom()
    exc1_per_atom = [exc1_per_atom[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    exc1_per_atom = cupy.asarray(exc1_per_atom)

    vhfopt = mf._opt_gpu.get(None, None)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        ej = rhf_grad._jk_energy_per_atom(
            mol, dm[0]+dm[1], vhfopt, with_k=False, verbose=verbose)[0]
        exc1_per_atom += ej
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        ej, ek = rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, verbose=verbose)
        ek *= hyb
        if omega != 0:
            vhfopt = mf._opt_gpu.get(omega, None)
            with mol.with_range_coulomb(omega):
                ek_lr = rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, with_j=False,
                                                     verbose=verbose)[1]
            ek += ek_lr * (alpha - hyb)

        exc1_per_atom += ej - ek

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

        vmat = cupy.zeros((nset,3,nao,nao))
        if xctype == 'LDA':
            ao_deriv = 1
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None, 
                                                         grid_range=(grid_start, grid_end)):
                mo_coeff_mask = mo_coeff[:,idx,:]
                rho_a = eval_rho2(_sorted_mol, ao_mask[0], mo_coeff_mask[0], mo_occ[0], None, xctype)
                rho_b = eval_rho2(_sorted_mol, ao_mask[0], mo_coeff_mask[1], mo_occ[1], None, xctype)

                vxc = ni.eval_xc_eff(xc_code, cupy.array([rho_a,rho_b]), 1, xctype=xctype)[1]
                wv = weight * vxc[:,0]
                aow = numint._scale_ao(ao_mask[0], wv[0])
                vtmp = rks_grad._d1_dot_(ao_mask[1:4], aow.T)
                add_sparse(vmat[0], vtmp, idx)
                aow = numint._scale_ao(ao_mask[0], wv[1])
                vtmp = rks_grad._d1_dot_(ao_mask[1:4], aow.T)
                add_sparse(vmat[1], vtmp, idx)
        elif xctype == 'GGA':
            ao_deriv = 2
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                mo_coeff_mask = mo_coeff[:,idx,:]
                rho_a = eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask[0], mo_occ[0], None, xctype)
                rho_b = eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask[1], mo_occ[1], None, xctype)

                vxc = ni.eval_xc_eff(xc_code, cupy.array([rho_a,rho_b]), 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[:,0] *= .5
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[0])
                add_sparse(vmat[0], vtmp, idx)
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[1])
                add_sparse(vmat[1], vtmp, idx)
        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 2
            for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                         grid_range=(grid_start, grid_end)):
                mo_coeff_mask = mo_coeff[:,idx,:]
                rho_a = eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask[0], mo_occ[0], None, xctype)
                rho_b = eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask[1], mo_occ[1], None, xctype)
                vxc = ni.eval_xc_eff(xc_code, cupy.array([rho_a,rho_b]), 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[:,0] *= .5
                wv[:,4] *= .5  # for the factor 1/2 in tau
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[0])
                vtmp += rks_grad._tau_grad_dot_(ao_mask, wv[0,4])
                add_sparse(vmat[0], vtmp, idx)
                vtmp = rks_grad._gga_grad_sum_(ao_mask, wv[1])
                vtmp += rks_grad._tau_grad_dot_(ao_mask, wv[1,4])
                add_sparse(vmat[1], vtmp, idx)

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
    dms = cupy.asarray(dms)
    dms = opt.sort_orbitals(dms, axis=[1,2])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

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

    # - sign because nabla_X = -nabla_x
    return exc, -cupy.array(vmat)

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
    assert dms.ndim == 3 and dms.shape[0] == 2
    dms = opt.sort_orbitals(dms.reshape(-1,nao0,nao0), axis=[1,2])

    excsum = cupy.zeros((natm, 3))
    vmat = cupy.zeros((2,3,nao,nao))
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

        for atm_id, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
            ngrids = weight.size
            for p0, p1 in lib.prange(0,ngrids,block_size):
                ao = numint.eval_ao(_sorted_mol, coords[p0:p1, :], ao_deriv, gdftopt=opt, transpose=False)
                if xctype == 'LDA':
                    rho_a = numint.eval_rho(_sorted_mol, ao[0], dms[0],
                                        xctype=xctype, hermi=1, with_lapl=False)
                    rho_b = numint.eval_rho(_sorted_mol, ao[0], dms[1],
                                        xctype=xctype, hermi=1, with_lapl=False)
                    rho = cupy.array([rho_a,rho_b])
                    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
                    exc = exc[:,0]
                else:
                    rho_a = numint.eval_rho(_sorted_mol, ao, dms[0],
                                        xctype=xctype, hermi=1, with_lapl=False)
                    rho_b = numint.eval_rho(_sorted_mol, ao, dms[1],
                                        xctype=xctype, hermi=1, with_lapl=False)
                    rho = cupy.array([rho_a,rho_b])
                    exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
                    exc = exc[:,0]

                if xctype == 'LDA':
                    wv = weight[p0:p1] * vxc[:,0]
                    aow = numint._scale_ao(ao[0], wv[0])
                    vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
                    rho = rho_a + rho_b
                    excsum += cupy.einsum('r,nxr->nx', exc*rho, weight1[:,:,p0:p1])
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms[0]) * 2
                    vmat[0] += vtmp
                    aow = numint._scale_ao(ao[0], wv[1])
                    vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
                    vmat[1] += vtmp
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms[1]) * 2
                    rho = vxc = aow = None

                elif xctype == 'GGA':
                    wv = weight[p0:p1] * vxc
                    wv[:,0] *= .5
                    vtmp = rks_grad._gga_grad_sum_(ao, wv[0])
                    vmat[0] += vtmp
                    rho = rho_a[0] + rho_b[0]
                    excsum += cupy.einsum('r,nxr->nx', exc*rho, weight1[:,:,p0:p1])
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms[0]) * 2
                    vtmp = rks_grad._gga_grad_sum_(ao, wv[1])
                    vmat[1] += vtmp
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms[1]) * 2
                    rho = vxc = None
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')

                elif xctype == 'MGGA':
                    wv = weight[p0:p1] * vxc
                    wv[:,0] *= .5
                    wv[:,4] *= .5

                    vtmp = rks_grad._gga_grad_sum_(ao, wv[0])
                    vtmp += rks_grad._tau_grad_dot_(ao, wv[0,4])
                    vmat[0] += vtmp
                    rho = rho_a[0] + rho_b[0]
                    excsum += cupy.einsum('r,nxr->nx', exc*rho, weight1[:,:,p0:p1])
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms[0]) * 2

                    vtmp = rks_grad._gga_grad_sum_(ao, wv[1])
                    vtmp += rks_grad._tau_grad_dot_(ao, wv[1,4])
                    vmat[1] += vtmp
                    excsum[atm_id] += cupy.einsum('xij,ji->x', vtmp, dms[1]) * 2
                    rho = vxc = None

    vmat = opt.unsort_orbitals(vmat, axis=[2,3])
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


def get_nlc_vxc(ni, mol, grids, xc_code, dms, mo_coeff, mo_occ, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)

    mol = None
    _sorted_mol = opt._sorted_mol
    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    mo_coeff_0 = opt.sort_orbitals(mo_coeff[0], axis=[0])
    mo_coeff_1 = opt.sort_orbitals(mo_coeff[1], axis=[0])
    nset = 1
    assert nset == 1

    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    ao_deriv = 2
    vvrho = []
    for ao_mask, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=max_memory):
        mo_coeff_mask_0 = mo_coeff_0[mask]
        mo_coeff_mask_1 = mo_coeff_1[mask]
        rhoa = eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask_0, mo_occ[0], None, xctype)
        rhob = eval_rho2(_sorted_mol, ao_mask[:4], mo_coeff_mask_1, mo_occ[1], None, xctype)
        vvrho.append(rhoa + rhob)
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
        vmat_tmp = rks_grad._gga_grad_sum_(ao_mask, wv)
        add_sparse(vmat, vmat_tmp, mask)

    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    exc = None
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


class Gradients(uhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device
    grid_response = False
    _keys = uks_grad.Gradients._keys

    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    get_veff = get_veff
    extra_force = rks_grad.Gradients.extra_force

Grad = Gradients
from gpu4pyscf import dft
dft.uks.UKS.Gradients = lib.class_as_method(Gradients)
