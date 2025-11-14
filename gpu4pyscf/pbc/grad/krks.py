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
from gpu4pyscf.pbc.grad import krhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2

__all__ = ['Gradients']

def get_veff(ks_grad, dm=None, kpts=None):
    mf = ks_grad.base
    cell = ks_grad.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    log = logger.new_logger(ks_grad)
    t0 = log.init_timer()

    if ks_grad.grid_response:
        raise NotImplementedError

    ni = mf._numint

    if isinstance(ni, multigrid.MultiGridNumInt):
        raise NotImplementedError(
            "Gradient with kpts not implemented with multigrid.MultiGridNumInt. "
            "Please use the default KNumInt or multigrid_v2.MultiGridNumInt instead.")

    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build()

    if kpts is None:
        nkpts = 1
    else:
        nkpts = len(kpts)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            exc = multigrid_v2.get_veff_ip1(ni, mf.xc, dm, with_j=True, with_pseudo_vloc_orbital_derivative=True, kpts=kpts).get()
            # exc of multigrid_v2 is the full response of dE/dX. However,
            # get_veff in grad_elec evaluates the contraction Tr(dm, <nabla|Veff|>).
            # They are differed by a factor of two. Scale exc to match the
            # convention of molecular rhf/rks get_veff.
            exc /= 2 * nkpts
        else:
            exc = get_vxc(ni, cell, grids, mf.xc, dm, kpts)
            t0 = log.timer('vxc', *t0)
            ej = ks_grad.get_j(dm, kpts)
            exc += ej
    else:
        from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
        with_rsjk = mf.rsjk
        if with_rsjk is None:
            raise NotImplementedError('Nuclear gradients for hybrid functional '
                                      'are only available via the rsjk method')
        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            exc = multigrid_v2.get_veff_ip1(ni, mf.xc, dm, with_j=True, with_pseudo_vloc_orbital_derivative=True, kpts=kpts).get()
            # exc of multigrid_v2 is the full response of dE/dX. However,
            # get_veff in grad_elec evaluates the contraction Tr(dm, <nabla|Veff|>).
            # They are differed by a factor of two. Scale exc to match the
            # convention of molecular rhf/rks get_veff.
            exc /= 2 * nkpts
            j_factor = 0
        else:
            exc = get_vxc(ni, cell, grids, mf.xc, dm, kpts)
            j_factor = 1
        omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)
        if omega != 0 and omega != with_rsjk.omega:
            with_rsjk = PBCJKMatrixOpt(cell, omega=omega).build()
        if with_rsjk.supmol is None:
            with_rsjk.build()
        exc += with_rsjk._get_ejk_sr_ip1(dm, kpts=kpts, exxdiv=mf.exxdiv,
                                         j_factor=j_factor, k_factor=k_sr)
        exc += with_rsjk._get_ejk_lr_ip1(dm, kpts=kpts, exxdiv=mf.exxdiv,
                                         j_factor=j_factor, k_factor=k_lr)
    return exc

def get_vxc(ni, cell, grids, xc_code, dm_kpts, kpts, hermi=1):
    '''derivatives of the Exc per cell'''
    assert dm_kpts.ndim == 3
    xctype = ni._xc_type(xc_code)
    nao = cell.nao
    nkpts = len(kpts)
    vmat = cp.zeros((nkpts,3,nao,nao), dtype=dm_kpts.dtype)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                                   sort_grids=True):
            rho = ni.eval_rho(cell, ao_ks[:,0], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
            wv = weight * vxc[0]
            aow = cp.einsum('kpi,p->kpi', ao_ks[:,0], wv)
            for kn in range(nkpts):
                vmat[kn] += _d1_dot_(ao_ks[kn,1:4], aow[kn])

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                                   sort_grids=True):
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
            wv = weight * vxc
            wv[0] *= .5
            for kn in range(nkpts):
                vmat[kn] += _gga_grad_sum_(ao_ks[kn], wv)

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                                   sort_grids=True):
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau
            for kn in range(nkpts):
                vmat[kn] += _gga_grad_sum_(ao_ks[kn], wv[:4])
                vmat[kn] += _tau_grad_dot_(ao_ks[kn], wv[4])

    elif xctype == 'HF':
        pass
    elif xctype == 'NLC':
        raise NotImplementedError("NLC")
    else:
        raise NotImplementedError(xc_code)

    aoslices = cell.aoslice_by_atom()
    exc = contract('kxij,kji->xi', vmat, dm_kpts).real.get()
    exc = np.array([exc[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    exc /= nkpts
    return -exc

def _d1_dot_(ao1, ao2, out=None):
    return rks_grad._d1_dot_(ao1.transpose(0,2,1), ao2)

def _gga_grad_sum_(ao, wv, out=None):
    return rks_grad._gga_grad_sum_(ao.transpose(0,2,1), wv)

def _tau_grad_dot_(ao, wv):
    return rks_grad._tau_grad_dot_(ao.transpose(0,2,1), wv)

class Gradients(rhf_grad.Gradients):
    _keys = {'grid_response', 'grids'}

    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    def reset(self, cell=None):
        if self.grids is not None:
            self.grids.reset(cell)
        return rhf_grad.Gradients.reset(self, cell)

    def dump_flags(self, verbose=None):
        rhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff

    def get_stress(self):
        from gpu4pyscf.pbc.grad import krks_stress
        return krks_stress.kernel(self)
