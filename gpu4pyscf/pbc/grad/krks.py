#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Analytical nuclear gradients for RKS with kpoints sampling
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.pbc.df import GDF
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.dft import multigrid, multigrid_v3, BeckeGrids
from gpu4pyscf.pbc.dft.gen_grid import get_becke_weight_derivative
from gpu4pyscf.pbc.dft.numint import _GTOvalOpt
from gpu4pyscf.pbc.grad.krks_stress import _eval_ao_strain_derivatives

__all__ = ['Gradients']

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9

def get_d2mu_dr2(ao_ks):
    assert ao_ks.ndim == 4
    nkpts = ao_ks.shape[0]
    ngrids = ao_ks.shape[2]
    nao = ao_ks.shape[3]

    d2mu_dr2 = cp.empty([nkpts, 3, 3, ngrids, nao], dtype = ao_ks.dtype)
    d2mu_dr2[:,0,0,:,:] = ao_ks[:,XX,:,:]
    d2mu_dr2[:,0,1,:,:] = ao_ks[:,XY,:,:]
    d2mu_dr2[:,1,0,:,:] = ao_ks[:,XY,:,:]
    d2mu_dr2[:,0,2,:,:] = ao_ks[:,XZ,:,:]
    d2mu_dr2[:,2,0,:,:] = ao_ks[:,XZ,:,:]
    d2mu_dr2[:,1,1,:,:] = ao_ks[:,YY,:,:]
    d2mu_dr2[:,1,2,:,:] = ao_ks[:,YZ,:,:]
    d2mu_dr2[:,2,1,:,:] = ao_ks[:,YZ,:,:]
    d2mu_dr2[:,2,2,:,:] = ao_ks[:,ZZ,:,:]
    return d2mu_dr2

def get_vxc(ni, cell, grids, xc_code, dm_kpts, kpts, hermi=1):
    '''derivatives of the Exc per cell'''
    assert dm_kpts.ndim == 3
    xctype = ni._xc_type(xc_code)
    nao = cell.nao
    nkpts = len(kpts)

    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 1
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")
    eval_gto_opt = _GTOvalOpt(cell, kpts, deriv=ao_deriv)

    vmat = cp.zeros((nkpts,3,nao,nao), dtype=dm_kpts.dtype)
    de_stress_rho = cp.zeros((3,3))
    exc_sum = 0

    if xctype == 'LDA':
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv + 1, kpts, sort_grids=True):
            rho = ni.eval_rho(cell, ao_ks[:,0], dm_kpts, xctype=xctype, hermi=hermi)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
            wv = weight * vxc[0]
            aow = cp.einsum('kpi,p->kpi', ao_ks[:,0], wv)
            for kn in range(nkpts):
                vmat[kn] += _d1_dot_(ao_ks[kn,1:4], aow[kn])

            del aow, vxc

            exc_sum += cp.sum(weight * (rho * exc))

            del rho, exc

            ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=ao_deriv, opt=eval_gto_opt)
            ao_ks_strain = ao_ks_strain[:,:,:,0]
            ao_ks_strain += contract('kxgp,yg->kxypg', ao_ks[:,1:4], coords.T)
            dm_nu = contract('kpq,kgq->kpg', dm_kpts, ao_ks[:,0].conj())
            drho_stress = contract('kxypg,kpg->xyg', ao_ks_strain, dm_nu)
            de_stress_rho += 2 * contract('xyg,g->xy', drho_stress, wv).real

            del ao_ks_strain, dm_nu, drho_stress, wv

    elif xctype == 'GGA':
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv + 1, kpts, sort_grids=True):
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
            wv = weight * vxc
            wv[0] *= .5
            for kn in range(nkpts):
                vmat[kn] += _gga_grad_sum_(ao_ks[kn], wv)

            del vxc

            exc_sum += cp.sum(weight * (rho[0] * exc))

            del rho, exc

            ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=ao_deriv, opt=eval_gto_opt)
            ao_ks_strain[:,:,:,0] += contract('kxgp,yg->kxypg', ao_ks[:,1:4], coords.T)
            d2ao = get_d2mu_dr2(ao_ks)
            ao_ks_strain[:,:,:,1:4] += contract('kxdgp,yg->kxydpg', d2ao, coords.T)
            del d2ao

            wv[0] *= 2
            dm_nu = contract('kpq,kgq->kpg', dm_kpts, ao_ks[:,0].conj())
            dmu_stress = contract('kxydpg,kpg->xydg', ao_ks_strain, dm_nu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dmu_stress, wv).real
            del dmu_stress, dm_nu
            dm_dmu = contract('kpq,kdgp->kdqg', dm_kpts, ao_ks[:,1:4])
            dnu_stress = contract('kxyqg,kdqg->xydg', ao_ks_strain[:,:,:,0].conj(), dm_dmu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dnu_stress, wv[1:4]).real
            del dnu_stress, dm_dmu

            del ao_ks_strain, wv

    elif xctype == 'MGGA':
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv + 1, kpts, sort_grids=True):
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau
            for kn in range(nkpts):
                vmat[kn] += _gga_grad_sum_(ao_ks[kn], wv[:4])
                vmat[kn] += _tau_grad_dot_(ao_ks[kn], wv[4])

            del vxc

            exc_sum += cp.sum(weight * (rho[0] * exc))

            del rho, exc

            ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=ao_deriv, opt=eval_gto_opt)
            ao_ks_strain[:,:,:,0] += contract('kxgp,yg->kxypg', ao_ks[:,1:4], coords.T)
            d2ao = get_d2mu_dr2(ao_ks)
            ao_ks_strain[:,:,:,1:4] += contract('kxdgp,yg->kxydpg', d2ao, coords.T)
            del d2ao

            wv[0] *= 2
            dm_nu = contract('kpq,kgq->kpg', dm_kpts, ao_ks[:,0].conj())
            dmu_stress = contract('kxydpg,kpg->xydg', ao_ks_strain, dm_nu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dmu_stress, wv[0:4]).real
            del dmu_stress, dm_nu
            dm_dmu = contract('kpq,kdgp->kdqg', dm_kpts, ao_ks[:,1:4])
            dnu_stress = contract('kxyqg,kdqg->xydg', ao_ks_strain[:,:,:,0].conj(), dm_dmu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dnu_stress, wv[1:4]).real
            del dnu_stress, dm_dmu
            dm_dnu = contract('kpq,kdgq->kdpg', dm_kpts, ao_ks[:,1:4].conj())
            tau_stress = contract('kxydpg,kdpg->xyg', ao_ks_strain[:,:,:,1:4], dm_dnu)
            de_stress_rho += 2 * contract('xyg,g->xy', tau_stress, wv[4]).real
            del tau_stress, dm_dnu

            del ao_ks_strain, wv

    elif xctype == 'HF':
        pass
    elif xctype == 'NLC':
        raise NotImplementedError("NLC")
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    de_stress_weight = exc_sum * cp.eye(3)

    exc = np.zeros((cell.natm + 3, 3))
    exc[:-3] = -krhf_grad.contract_h1e_dm(cell, vmat, dm_kpts, hermi=1)
    exc[:-3] *= 1.0 / nkpts
    exc[-3:] = (de_stress_rho / nkpts + de_stress_weight).get()
    return exc

def get_vxc_full_response(ni, cell, grids, xc_code, dm_kpts, kpts, hermi=1):
    ''' dExc/dR for Becke grids, where grid response is included '''
    # TODO: apply sparsity in ao_ks, remove zero-weight grids
    assert isinstance(grids, BeckeGrids)
    assert dm_kpts.ndim == 3
    assert hermi == 1, "Only hermitian dm_kpts is supported, otherwise grid density is not real, and we're not able to evaluate xc functional."
    xctype = ni._xc_type(xc_code)
    nao = cell.nao
    natm = cell.natm
    nkpts = len(kpts)
    ngrids = grids.coords.shape[0]

    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 1
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")
    eval_gto_opt = _GTOvalOpt(cell, kpts, deriv=ao_deriv)

    de_grid_response_weight = cp.zeros((natm + 3, 3), dtype=cp.float64)
    g1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts):
        g0, g1 = g1, g1 + weight.size
        rho = ni.eval_rho(cell, ao_ks, dm_kpts, xctype=xctype, hermi=hermi)
        exc = ni.eval_xc_eff(xc_code, rho, deriv=0, xctype=xctype, spin=0)[0]
        if rho.ndim == 2:
            rho = rho[0]
        else:
            assert rho.ndim == 1
        dweight_dA = get_becke_weight_derivative(grids, natm, (g0,g1))
        de_grid_response_weight += cp.einsum("Adg->Ad", dweight_dA * (rho * exc))
        del dweight_dA, rho, exc
    assert g1 == ngrids

    dvmat_orbital_response = cp.zeros((nkpts,3,nao,nao), dtype=dm_kpts.dtype)
    de_grid_response_rho = cp.zeros((natm, 3), dtype=dm_kpts.dtype)
    de_stress_rho = cp.zeros((3,3), dtype=cp.float64)

    g1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv + 1, kpts):
        g0, g1 = g1, g1 + weight.size

        i_atom = int(grids.supatm_to_atm_idx[grids.supatm_idx[g0]])
        assert cp.max(cp.abs(grids.supatm_to_atm_idx[grids.supatm_idx[g0:g1]] - i_atom)) == 0 # Guaranteed by get_becke_grids()

        if xctype == 'LDA':
            rho = ni.eval_rho(cell, ao_ks[:,0], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[1]

            wv = weight * vxc[0]
            aow = cp.einsum('kpi,p->kpi', ao_ks[:,0], wv)
            for kn in range(nkpts):
                vtmp_k = _d1_dot_(ao_ks[kn,1:4], aow[kn])
                dvmat_orbital_response[kn] += vtmp_k
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_k, dm_kpts[kn]) * 2
                del vtmp_k

            del aow, rho, vxc

            ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=ao_deriv, opt=eval_gto_opt)
            ao_ks_strain = ao_ks_strain[:,:,:,0]
            associated_supatm_coords = grids.supatm_coords[grids.supatm_idx[g0:g1]]
            ao_ks_strain += contract('kxgp,yg->kxypg', ao_ks[:,1:4], associated_supatm_coords.T)
            del associated_supatm_coords

            dm_nu = contract('kpq,kgq->kpg', dm_kpts, ao_ks[:,0].conj())
            drho_stress = contract('kxypg,kpg->xyg', ao_ks_strain, dm_nu)
            de_stress_rho += 2 * contract('xyg,g->xy', drho_stress, wv).real

            del ao_ks_strain, dm_nu, drho_stress, wv

        elif xctype == 'GGA':
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[1]

            wv = weight * vxc
            wv[0] *= .5
            for kn in range(nkpts):
                vtmp_k = _gga_grad_sum_(ao_ks[kn], wv)
                dvmat_orbital_response[kn] += vtmp_k
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_k, dm_kpts[kn]) * 2
                del vtmp_k
            del rho, vxc

            ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=ao_deriv, opt=eval_gto_opt)
            associated_supatm_coords = grids.supatm_coords[grids.supatm_idx[g0:g1]]
            ao_ks_strain[:,:,:,0] += contract('kxgp,yg->kxypg', ao_ks[:,1:4], associated_supatm_coords.T)
            d2ao = get_d2mu_dr2(ao_ks)
            ao_ks_strain[:,:,:,1:4] += contract('kxdgp,yg->kxydpg', d2ao, associated_supatm_coords.T)
            del d2ao, associated_supatm_coords

            wv[0] *= 2
            dm_nu = contract('kpq,kgq->kpg', dm_kpts, ao_ks[:,0].conj())
            dmu_stress = contract('kxydpg,kpg->xydg', ao_ks_strain, dm_nu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dmu_stress, wv).real
            del dmu_stress, dm_nu
            dm_dmu = contract('kpq,kdgp->kdqg', dm_kpts, ao_ks[:,1:4])
            dnu_stress = contract('kxyqg,kdqg->xydg', ao_ks_strain[:,:,:,0].conj(), dm_dmu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dnu_stress, wv[1:4]).real
            del dnu_stress, dm_dmu

            del ao_ks_strain, wv

        elif xctype == 'MGGA':
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[1]

            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau
            for kn in range(nkpts):
                vtmp_k = _gga_grad_sum_(ao_ks[kn], wv[:4]) + _tau_grad_dot_(ao_ks[kn], wv[4])
                dvmat_orbital_response[kn] += vtmp_k
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_k, dm_kpts[kn]) * 2
                del vtmp_k
            del rho, vxc

            ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=ao_deriv, opt=eval_gto_opt)
            associated_supatm_coords = grids.supatm_coords[grids.supatm_idx[g0:g1]]
            ao_ks_strain[:,:,:,0] += contract('kxgp,yg->kxypg', ao_ks[:,1:4], associated_supatm_coords.T)
            d2ao = get_d2mu_dr2(ao_ks)
            ao_ks_strain[:,:,:,1:4] += contract('kxdgp,yg->kxydpg', d2ao, associated_supatm_coords.T)
            del d2ao, associated_supatm_coords

            wv[0] *= 2
            dm_nu = contract('kpq,kgq->kpg', dm_kpts, ao_ks[:,0].conj())
            dmu_stress = contract('kxydpg,kpg->xydg', ao_ks_strain, dm_nu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dmu_stress, wv[0:4]).real
            del dmu_stress, dm_nu
            dm_dmu = contract('kpq,kdgp->kdqg', dm_kpts, ao_ks[:,1:4])
            dnu_stress = contract('kxyqg,kdqg->xydg', ao_ks_strain[:,:,:,0].conj(), dm_dmu)
            de_stress_rho += 2 * contract('xydg,dg->xy', dnu_stress, wv[1:4]).real
            del dnu_stress, dm_dmu
            dm_dnu = contract('kpq,kdgq->kdpg', dm_kpts, ao_ks[:,1:4].conj())
            tau_stress = contract('kxydpg,kdpg->xyg', ao_ks_strain[:,:,:,1:4], dm_dnu)
            de_stress_rho += 2 * contract('xyg,g->xy', tau_stress, wv[4]).real
            del tau_stress, dm_dnu

            del ao_ks_strain, wv

        else:
            raise NotImplementedError(f"Unrecognized xctype = {xctype}")
    assert g1 == ngrids

    exc = np.zeros((cell.natm + 3, 3), dtype=np.float64)
    exc[:-3] = de_grid_response_rho.get().real
    exc[:-3] -= krhf_grad.contract_h1e_dm(cell, dvmat_orbital_response, dm_kpts, hermi=1)
    exc[-3:] = de_stress_rho.get()
    exc *= 1.0 / nkpts
    exc += de_grid_response_weight.get()
    return exc

def _d1_dot_(ao1, ao2, out=None):
    return rks_grad._d1_dot_(ao1.transpose(0,2,1), ao2)

def _gga_grad_sum_(ao, wv, out=None):
    return rks_grad._gga_grad_sum_(ao.transpose(0,2,1), wv)

def _tau_grad_dot_(ao, wv):
    return rks_grad._tau_grad_dot_(ao.transpose(0,2,1), wv)


class Gradients(krhf_grad.Gradients):

    def reset(self, cell=None):
        if self.grids is not None:
            self.grids.reset(cell)
        return krhf_grad.Gradients.reset(self, cell)

    def dump_flags(self, verbose=None):
        krhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    def energy_ee(self, dm, kpts):
        mf = self.base
        log = logger.new_logger(self)
        t0 = log.init_timer()

        ni = mf._numint
        xc = getattr(mf, 'xc', 'HF')
        if xc.upper() == 'HF':
            omega, k_lr, k_sr = 0, 1, 1
        else:
            omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)
        j_factor = 1

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        de = np.zeros([self.cell.natm+3, 3])
        if isinstance(ni, multigrid_v3.MultiGridNumInt):
            de = ni.energy_derivatives(
                xc, dm, kpts=kpts, spin=0, with_j=True, with_nuc=True)
            j_factor = 0
        elif isinstance(ni, multigrid.MultiGridNumIntBase):
            raise NotImplementedError(f'derivatives for {ni}')
        else:
            grids = self.grids or mf.grids
            if grids.coords is None:
                grids.build()
            cell = self.cell
            if self.grid_response:
                assert isinstance(grids, BeckeGrids), "Only Becke grid requires grid response"
                fn = get_vxc_full_response
            else:
                fn = get_vxc
            de = fn(ni, cell, grids, xc, dm, kpts)
        t0 = log.timer_debug1('vxc', *t0)

        if j_factor != 0 or k_sr != 0 or k_lr != 0:
            de += krhf_grad._get_ejk_derivatives(
                mf, dm, kpts, j_factor, omega, k_lr, k_sr)
            t0 = log.timer_debug1('JK', *t0)
        return de
