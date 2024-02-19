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

import ctypes
import numpy as np
import cupy
import pyscf
from pyscf import lib
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem, add_sparse, load_library, take_last2d, tag_array
from gpu4pyscf.lib import logger
from pyscf import __config__

MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 128*128)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)

libgdft = load_library('libgdft')
libgdft.GDFT_make_dR_dao_w.restype = ctypes.c_int

# TODO: there are many get_jk, which can be replaced by get_j or get_k!

def get_veff(ks_grad, mol=None, dm=None):
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
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(sort_grids=True)

    ni = mf._numint
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc_tmp = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            raise NotImplementedError
    else:
        exc, vxc_tmp = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            dma =  dm[0]
            dma = tag_array(dma, mo_coeff=mf.mo_coeff[0], mo_occ=mf.mo_occ[0])
            dmb =  dm[1]
            dmb = tag_array(dmb, mo_coeff=mf.mo_coeff[1], mo_occ=mf.mo_occ[1])
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dma,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc_tmp[0] += vnlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dmb,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc_tmp[1] += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    mo_coeff_alpha = mf.mo_coeff[0]
    mo_coeff_beta = mf.mo_coeff[1]
    occ_coeff0 = cupy.asarray(mo_coeff_alpha[:, mf.mo_occ[0]>0.5], order='C')
    occ_coeff1 = cupy.asarray(mo_coeff_beta[:, mf.mo_occ[1]>0.5], order='C')
    # print(mf.mo_coeff.shape)
    # print(mf.mo_occ.shape)
    # print(mf.mo_coeff[1, :, mf.mo_occ[1]>0.5].shape)
    # print(vxc_tmp.shape, occ_coeff0.shape, type(vxc_tmp), type(occ_coeff0))
    tmp = contract('nij,jk->nik', vxc_tmp[0], occ_coeff0)
    vxc = contract('nik,ik->ni', tmp, occ_coeff0)
    tmp = contract('nij,jk->nik', vxc_tmp[1], occ_coeff1)
    vxc+= contract('nik,ik->ni', tmp, occ_coeff1)

    aoslices = mol.aoslice_by_atom()
    vxc = [vxc[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    vxc = cupy.asarray(vxc)
    
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm[0]+dm[1])
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vk0 = ks_grad.get_jk(mol, dm[0])[1]
        vk1 = ks_grad.get_jk(mol, dm[1])[1]
        vj0 = ks_grad.get_jk(mol, dm[0]+dm[1])[0]
        vk = (vk0+vk1) * hyb
        if omega != 0:
            vk_lr0 = ks_grad.get_k(mol, dm[0], omega=omega) 
            vk_lr1 = ks_grad.get_k(mol, dm[1], omega=omega) 
            vk += (vk_lr0+vk_lr1) * (alpha - hyb)
            
        vxc += vj0 - vk

    return vxc


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mo_occ = cupy.asarray(dms.mo_occ)
    mo_coeff = cupy.asarray(dms.mo_coeff)
    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms)
    dms = take_last2d(dms, opt.ao_idx)
    # dms = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff)
    #        for dm in dms.reshape(-1,nao0,nao0)]
    # mo_coeff = cupy.einsum('pq,sqt->spt',coeff,mo_coeff)
    # mo_coeff = coeff @ mo_coeff
    mo_coeff = mo_coeff[:, opt.ao_idx]

    nset = len(dms)
    vmat = cupy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao_mask, idx, weight, _ in ni.block_loop(opt.mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[:,idx,:]
            rho_a = numint.eval_rho2(opt.mol, ao_mask[0], mo_coeff_mask[0], mo_occ[0], None, xctype)
            rho_b = numint.eval_rho2(opt.mol, ao_mask[0], mo_coeff_mask[1], mo_occ[1], None, xctype)
            
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
        for ao_mask, idx, weight, _ in ni.block_loop(opt.mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[:,idx,:]
            rho_a = numint.eval_rho2(opt.mol, ao_mask[:4], mo_coeff_mask[0], mo_occ[0], None, xctype)
            rho_b = numint.eval_rho2(opt.mol, ao_mask[:4], mo_coeff_mask[1], mo_occ[1], None, xctype)
            
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
        for ao_mask, idx, weight, _ in ni.block_loop(opt.mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[:,idx,:]
            rho_a = numint.eval_rho2(opt.mol, ao_mask[:10], mo_coeff_mask[0], mo_occ[0], None, xctype, with_lapl=False)
            rho_b = numint.eval_rho2(opt.mol, ao_mask[:10], mo_coeff_mask[1], mo_occ[1], None, xctype, with_lapl=False)
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
    # vmat = [cupy.einsum('pi,npq,qj->nij', coeff, v, coeff) for v in vmat]
    vmat = take_last2d(vmat, opt.rev_ao_idx)
    exc = None
    if nset == 1:
        vmat = vmat[0]

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
    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms)
    dms = take_last2d(dms, opt.ao_idx)
    # dms = [cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff)
    #        for dm in dms.reshape(-1,nao0,nao0)]
    # mo_coeff = cupy.einsum('pq,sqt->spt',coeff,mo_coeff)
    # mo_coeff = coeff @ mo_coeff
    
    excsum = 0
    vmat = cupy.zeros((2,3,nao,nao))
    with opt.gdft_envs_cache():
        if xctype == 'LDA':
            ao_deriv = 1
        else:
            ao_deriv = 2

        mem_avail = get_avail_mem()
        comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
        block_size = int((mem_avail*.4/8/(comp+1)/nao - 3*nao*2)/ ALIGNED) * ALIGNED
        block_size = min(block_size//2, MIN_BLK_SIZE)
        log.debug1('Available GPU mem %f Mb, block_size %d', mem_avail/1e6, block_size)

        if block_size < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        for atm_id, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
            ngrids = weight.size
            for p0, p1 in lib.prange(0,ngrids,block_size):
                ao = numint.eval_ao(ni, opt.mol, coords[p0:p1, :], ao_deriv)
                if xctype == 'LDA':
                    rho_a = numint.eval_rho(opt.mol, ao, dms[0],
                                        xctype='GGA', hermi=1, with_lapl=False)
                    rho_b = numint.eval_rho(opt.mol, ao, dms[1],
                                        xctype='GGA', hermi=1, with_lapl=False)
                    vxc = ni.eval_xc_eff(xc_code, cupy.array([rho_a[0],rho_b[0]]), 1, xctype=xctype)[1]
                else:
                    rho_a = numint.eval_rho(opt.mol, ao, dms[0],
                                        xctype=xctype, hermi=1, with_lapl=False)
                    rho_b = numint.eval_rho(opt.mol, ao, dms[1],
                                        xctype=xctype, hermi=1, with_lapl=False)
                    vxc = ni.eval_xc_eff(xc_code, cupy.array([rho_a,rho_b]), 1, xctype=xctype)[1]

                if xctype == 'LDA':
                    wv = weight[p0:p1] * vxc[:,0]
                    aow = numint._scale_ao(ao[0], wv[0])
                    vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
                    vmat[0] += vtmp
                    aow = numint._scale_ao(ao[0], wv[1])
                    vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
                    vmat[1] += vtmp

                elif xctype == 'GGA':
                    wv = weight[p0:p1] * vxc
                    wv[:,0] *= .5
                    vtmp = rks_grad._gga_grad_sum_(ao, wv[0])
                    vmat[0] += vtmp
                    vtmp = rks_grad._gga_grad_sum_(ao, wv[1])
                    vmat[1] += vtmp
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')

                elif xctype == 'MGGA':
                    wv = weight[p0:p1] * vxc
                    wv[:,0] *= .5
                    wv[:,4] *= .5

                    vtmp = rks_grad._gga_grad_sum_(ao, wv[0])
                    vtmp += rks_grad._tau_grad_dot_(ao, wv[0,4])
                    vmat[0] += vtmp

                    vtmp = rks_grad._gga_grad_sum_(ao, wv[1])
                    vtmp += rks_grad._tau_grad_dot_(ao, wv[1,4])
                    vmat[1] += vtmp

    excsum = None
    vmat = take_last2d(vmat, opt.rev_ao_idx)
    # vmat = cupy.einsum('pi,npq,qj->nij', coeff, vmat, coeff)

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


class Gradients(uhf_grad.Gradients, pyscf.grad.uks.Gradients):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device
    
    get_veff = get_veff
    
    def get_dispersion(self):
        if self.base.disp[:2].upper() == 'D3':
            from pyscf import lib
            with lib.with_omp_threads(1):
                import dftd3.pyscf as disp
                d3 = disp.DFTD3Dispersion(self.mol, xc=self.base.xc, version=self.base.disp)
                _, g_d3 = d3.kernel()
            return g_d3

        if self.base.disp[:2].upper() == 'D4':
            from pyscf.data.elements import charge
            atoms = np.array([ charge(a[0]) for a in self.mol._atom])
            coords = self.mol.atom_coords()

            from pyscf import lib
            with lib.with_omp_threads(1):
                from dftd4.interface import DampingParam, DispersionModel
                model = DispersionModel(atoms, coords)
                res = model.get_dispersion(DampingParam(method=self.base.xc), grad=True)
            return res.get("gradient")