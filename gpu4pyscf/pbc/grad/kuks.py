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
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.pbc.grad import kuhf as kuhf_grad
from gpu4pyscf.pbc.grad import krks as krks_grad
from gpu4pyscf.lib.cupy_helper import contract

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
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build()

    exc = get_vxc(ni, cell, grids, mf.xc, dm, kpts)
    t0 = log.timer('vxc', *t0)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        ej = ks_grad.get_j(dm[0]+dm[1], kpts)
        exc += ej
    else:
        raise NotImplementedError
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        ej = mf.get_j(dm[0]+dm[1], kpts)
        ek = mf.get_k(dm, kpts)
        ek *= hyb
        if omega != 0:
            with cell.with_range_coulomb(omega):
                ek += ks_grad.get_k(dm, kpts) * (alpha - hyb)
        exc += ej - ek

    return exc

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
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
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
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
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
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
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

    aoslices = cell.aoslice_by_atom()
    exc = contract('skxij,skji->xi', vmat, dm_kpts).real.get()
    exc = np.array([exc[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    return -exc

class Gradients(kuhf_grad.Gradients):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    _keys = {'grid_response', 'grids'}

    def __init__(self, mf):
        kuhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    reset = krks_grad.Gradients.reset
    dump_flags = krks_grad.Gradients.dump_flags

    get_veff = get_veff

    def get_stress(self):
        from gpu4pyscf.pbc.grad import kuks_stress
        return kuks_stress.kernel(self)
