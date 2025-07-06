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
from gpu4pyscf.pbc.dft import UniformGrids
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import contract


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

    assert isinstance(grids, UniformGrids)
    if grids.coords is None:
        grids.build()

    exc = get_vxc(ni, cell, grids, mf.xc, dm, kpts)
    t0 = logger.timer(ks_grad, 'vxc', *t0)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        ej = ks_grad.get_j(dm, kpts)
        exc += ej
    else:
        raise NotImplementedError
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        ej, ek = ks_grad.get_jk(dm, kpts)
        ek *= hyb
        if omega != 0:
            with cell.with_range_coulomb(omega):
                ek += ks_grad.get_k(dm, kpts) * (alpha - hyb)
        vxc += ej - ek * .5
    return exc

def get_vxc(ni, cell, grids, xc_code, dm_kpts, kpts, kpts_band=None, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    nkpts = len(kpts)
    vmat = cp.zeros((nkpts,3,nao,nao), dtype=dm_kpts.dtype)
    assert dm_kpts.ndim == 3
    if xctype in (None, 'LDA', 'HF'):
        ao_deriv = 1
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts):
            rho = ni.eval_rho(cell, ao_ks[:,0], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
            wv = weight * vxc[0]
            aow = cp.einsum('kpi,p->kpi', ao_ks[:,0], wv)
            ao_ks = ao_ks.transpose(0,1,3,2)
            for kn in range(nkpts):
                vmat[kn] += rks_grad._d1_dot_(ao_ks[kn,1:4], aow[kn])
            rho = vrho = aow = None

    elif xctype=='GGA':
        ao_deriv = 2
        for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts):
            rho = ni.eval_rho(cell, ao_ks[:,:4], dm_kpts, xctype=xctype, hermi=hermi)
            vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
            wv = weight * vxc
            wv[0] *= .5
            ao_ks = ao_ks.transpose(0,1,3,2)
            for kn in range(nkpts):
                vmat[kn] += rks_grad._gga_grad_sum_(ao_ks[kn], wv)
            rho = vxc = wv = None

    elif xctype=='NLC':
        raise NotImplementedError("NLC")
    else:
        raise NotImplementedError("metaGGA")

    aoslices = cell.aoslice_by_atom()
    exc = contract('kxij,kji->xi', vmat, dm_kpts).real.get()
    exc = np.array([exc[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    return -exc

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
