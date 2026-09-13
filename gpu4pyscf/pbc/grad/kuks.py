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
Analytical nuclear gradients for UKS with kpoints sampling
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.pbc.grad import kuhf as kuhf_grad
from gpu4pyscf.pbc.grad import krks as krks_grad
from gpu4pyscf.pbc.df import GDF
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.dft import multigrid, BeckeGrids
from gpu4pyscf.pbc.dft.gen_grid import get_becke_weight_derivative

__all__ = ['Gradients']

def get_vxc(ni, cell, grids, xc_code, dm_kpts, kpts, hermi=1):
    assert dm_kpts.ndim == 4
    xctype = ni._xc_type(xc_code)
    nao = cell.nao
    nkpts = len(kpts)
    vmat = cp.zeros((2,nkpts,3,nao,nao), dtype=dm_kpts.dtype)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                                   sort_grids=True):
            rho_a = ni.eval_rho(cell, ao_ks[:,0], dm_kpts[0], xctype=xctype, hermi=hermi)
            rho_b = ni.eval_rho(cell, ao_ks[:,0], dm_kpts[1], xctype=xctype, hermi=hermi)
            rho = cp.stack([rho_a, rho_b], axis=0)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[1]
            wv = weight * vxc[:,0]
            aowa = cp.einsum('xpi,p->xpi', ao_ks[:,0], wv[0])
            aowb = cp.einsum('xpi,p->xpi', ao_ks[:,0], wv[1])
            for kn in range(nkpts):
                vmat[0,kn] += krks_grad._d1_dot_(ao_ks[kn,1:4], aowa[kn])
                vmat[1,kn] += krks_grad._d1_dot_(ao_ks[kn,1:4], aowb[kn])

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                                   sort_grids=True):
            rho_a = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[0], xctype=xctype, hermi=hermi)
            rho_b = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[1], xctype=xctype, hermi=hermi)
            rho = cp.stack([rho_a, rho_b], axis=0)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            for kn in range(nkpts):
                vmat[0,kn] += krks_grad._gga_grad_sum_(ao_ks[kn], wv[0])
                vmat[1,kn] += krks_grad._gga_grad_sum_(ao_ks[kn], wv[1])

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                                   sort_grids=True):
            rho_a = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[0], xctype=xctype, hermi=hermi)
            rho_b = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[1], xctype=xctype, hermi=hermi)
            rho = cp.stack([rho_a, rho_b], axis=0)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5  # for the factor 1/2 in tau
            for kn in range(nkpts):
                vmat[0,kn] += krks_grad._gga_grad_sum_(ao_ks[kn], wv[0,:4])
                vmat[1,kn] += krks_grad._gga_grad_sum_(ao_ks[kn], wv[1,:4])
                vmat[0,kn] += krks_grad._tau_grad_dot_(ao_ks[kn], wv[0,4])
                vmat[1,kn] += krks_grad._tau_grad_dot_(ao_ks[kn], wv[1,4])

    elif xctype == 'HF':
        pass
    elif xctype == 'NLC':
        raise NotImplementedError("NLC")
    else:
        raise NotImplementedError(xc_code)

    exc = krhf_grad.contract_h1e_dm(cell, vmat, dm_kpts, hermi=1)
    exc *= -1.0 / nkpts
    return exc

def get_vxc_full_response(ni, cell, grids, xc_code, dm_kpts, kpts, hermi=1):
    ''' dExc/dR for Becke grids, where grid response is included '''
    # TODO: apply sparsity in ao_ks, remove zero-weight grids
    assert isinstance(grids, BeckeGrids)
    assert dm_kpts.ndim == 4
    assert dm_kpts.shape[0] == 2
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

    de_grid_response_weight = cp.zeros((natm, 3), dtype=cp.float64)
    g1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts):
        g0, g1 = g1, g1 + weight.size
        rho_a = ni.eval_rho(cell, ao_ks, dm_kpts[0], xctype=xctype, hermi=hermi)
        rho_b = ni.eval_rho(cell, ao_ks, dm_kpts[1], xctype=xctype, hermi=hermi)
        rho = cp.stack([rho_a, rho_b], axis=0)
        del rho_a, rho_b
        exc = ni.eval_xc_eff(xc_code, rho, deriv=0, xctype=xctype, spin=1)[0]
        rho = rho[0] + rho[1]
        if rho.ndim == 2:
            rho = rho[0]
        else:
            assert rho.ndim == 1
        dweight_dA = get_becke_weight_derivative(grids, natm, (g0,g1))
        de_grid_response_weight += cp.einsum("Adg->Ad", dweight_dA * (rho * exc))
        del dweight_dA, rho, exc
    assert g1 == ngrids

    dvmat_orbital_response = cp.zeros((2,nkpts,3,nao,nao), dtype=dm_kpts.dtype)
    de_grid_response_rho = cp.zeros((natm, 3), dtype=dm_kpts.dtype)

    g1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv + 1, kpts):
        g0, g1 = g1, g1 + weight.size

        i_atom = int(grids.supatm_to_atm_idx[grids.supatm_idx[g0]])
        assert cp.max(cp.abs(grids.supatm_to_atm_idx[grids.supatm_idx[g0:g1]] - i_atom)) == 0 # Guaranteed by get_becke_grids()

        if xctype == 'LDA':
            rho_a = ni.eval_rho(cell, ao_ks[:,0], dm_kpts[0], xctype=xctype, hermi=hermi)
            rho_b = ni.eval_rho(cell, ao_ks[:,0], dm_kpts[1], xctype=xctype, hermi=hermi)
            rho = cp.stack([rho_a, rho_b], axis=0)
            del rho_a, rho_b
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[1]
            wv = weight * vxc[:,0]
            aowa = cp.einsum('kpi,p->kpi', ao_ks[:,0], wv[0])
            aowb = cp.einsum('kpi,p->kpi', ao_ks[:,0], wv[1])
            for kn in range(nkpts):
                vtmp_a = krks_grad._d1_dot_(ao_ks[kn,1:4], aowa[kn])
                dvmat_orbital_response[0,kn] += vtmp_a
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_a, dm_kpts[0,kn]) * 2
                vtmp_b = krks_grad._d1_dot_(ao_ks[kn,1:4], aowb[kn])
                dvmat_orbital_response[1,kn] += vtmp_b
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_b, dm_kpts[1,kn]) * 2
                del vtmp_a, vtmp_b
            del wv, rho, aowa, aowb, vxc

        elif xctype == 'GGA':
            rho_a = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[0], xctype=xctype, hermi=hermi)
            rho_b = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[1], xctype=xctype, hermi=hermi)
            rho = cp.stack([rho_a, rho_b], axis=0)
            del rho_a, rho_b
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            for kn in range(nkpts):
                vtmp_a = krks_grad._gga_grad_sum_(ao_ks[kn], wv[0])
                dvmat_orbital_response[0,kn] += vtmp_a
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_a, dm_kpts[0,kn]) * 2
                vtmp_b = krks_grad._gga_grad_sum_(ao_ks[kn], wv[1])
                dvmat_orbital_response[1,kn] += vtmp_b
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_b, dm_kpts[1,kn]) * 2
                del vtmp_a, vtmp_b
            del wv, rho, vxc

        elif xctype == 'MGGA':
            rho_a = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[0], xctype=xctype, hermi=hermi)
            rho_b = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts[1], xctype=xctype, hermi=hermi)
            rho = cp.stack([rho_a, rho_b], axis=0)
            del rho_a, rho_b
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5  # for the factor 1/2 in tau
            for kn in range(nkpts):
                vtmp_a = krks_grad._gga_grad_sum_(ao_ks[kn], wv[0,:4]) + krks_grad._tau_grad_dot_(ao_ks[kn], wv[0,4])
                dvmat_orbital_response[0,kn] += vtmp_a
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_a, dm_kpts[0,kn]) * 2
                vtmp_b = krks_grad._gga_grad_sum_(ao_ks[kn], wv[1,:4]) + krks_grad._tau_grad_dot_(ao_ks[kn], wv[1,4])
                dvmat_orbital_response[1,kn] += vtmp_b
                de_grid_response_rho[i_atom] += cp.einsum('xij,ji->x', vtmp_b, dm_kpts[1,kn]) * 2
                del vtmp_a, vtmp_b
            del wv, rho, vxc

        else:
            raise NotImplementedError(f"Unrecognized xctype = {xctype}")
    assert g1 == ngrids

    exc = de_grid_response_rho.get().real
    exc -= krhf_grad.contract_h1e_dm(cell, dvmat_orbital_response, dm_kpts, hermi=1)
    exc *= 1.0 / nkpts
    exc += de_grid_response_weight.get()
    return exc

class Gradients(kuhf_grad.Gradients):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    grids = None
    grid_response = False

    _keys = {'grid_response', 'grids'}

    reset = krks_grad.Gradients.reset
    dump_flags = krks_grad.Gradients.dump_flags

    def energy_ee(self, dm, kpts):
        mf = self.base
        with_df = mf.with_df
        log = logger.new_logger(self)
        t0 = log.init_timer()

        if self.grid_response:
            raise NotImplementedError

        if isinstance(mf.grids, BeckeGrids):
            raise NotImplementedError('gradients for BeckeGrids not supported')

        ni = mf._numint
        j_in_xc = not isinstance(with_df, GDF)
        j_factor = 1
        if j_in_xc:
            j_factor = 0
        xc = getattr(mf, 'xc', 'HF')
        if xc.upper() == 'HF':
            omega, k_lr, k_sr = 0, 1, 1
        else:
            omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        de = 0
        if isinstance(ni, multigrid.MultiGridNumIntBase):
            de = ni.energy_derivatives(
                xc, dm, kpts=kpts, spin=1, with_j=j_in_xc, with_nuc=True)
        else:
            if self.grids is not None:
                grids = self.grids
            else:
                grids = mf.grids
            if grids.coords is None:
                grids.build()
            if ks_grad.grid_response:
                assert isinstance(grids, BeckeGrids), "Only Becke grid requires grid response"
                fn = get_vxc_full_response
            else:
                fn = get_vxc
            cell = self.cell
            de = np.empty([cell.natm+3, 3])
            de[:-3] = fn(ni, cell, grids, xc, dm, kpts)
            de[-3:] = np.nan
        t0 = log.timer_debug1('vxc', *t0)

        if j_factor != 0 or k_sr != 0 or k_lr != 0:
            de += krhf_grad._get_ejk_derivatives(
                mf, dm, kpts, j_factor, omega, k_lr, k_sr)
            t0 = log.timer_debug1('JK', *t0)
        return de
