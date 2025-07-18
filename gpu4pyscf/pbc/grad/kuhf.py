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
Analytical nuclear gradients for RHF with kpoints sampling
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.lib.cupy_helper import contract, ensure_numpy

__all__ = ['Gradients']

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    cell = mf_grad.cell
    natm = cell.natm
    kpts = mf.kpts
    nkpts = len(kpts)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    if getattr(mf, 'disp', None):
        raise NotImplementedError('dispersion correction')

    log = logger.new_logger(mf_grad)
    t0 = log.init_timer()
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    hcore_deriv = mf_grad.hcore_generator(cell, kpts)
    s1 = mf_grad.get_ovlp(cell, kpts)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dvhf = mf_grad.get_veff(dm0, kpts)
    t1 = log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]
    aoslices = cell.aoslice_by_atom()
    extra_force = np.empty([natm, 3])
    dh1e = cp.empty([natm, 3])
    for ia in range(natm):
        h1ao = hcore_deriv(ia)
        dh1e[ia] = cp.einsum('kxij,kji->x', h1ao, dm0_sf).real
        extra_force[ia] = ensure_numpy(mf_grad.extra_force(ia, locals()))
    log.timer('gradients of 1e part', *t1)

    # nabla is applied on bra in vhf. *2 for the contributions of nabla|ket>
    ds = contract('kxij,kji->xi', s1, dme0_sf).real
    ds = (-2 * ds).get()
    ds = np.array([ds[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    de = (2 * dvhf + dh1e.get() + ds) / nkpts + extra_force

    if log.verbose > logger.DEBUG:
        log.debug('gradients of electronic part')
        mf_grad._write(cell, de, atmlst)
    return de

class Gradients(krhf_grad.GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    def get_veff(self, dm=None, kpts=None):
        if dm is None: dm = self.base.make_rdm1()
        ej = self.get_j(dm[0]+dm[1], kpts)
        ek = self.get_k(dm, kpts)
        return ej - ek

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        '''Energy weighted density matrix'''
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        dm1ea = krhf_grad.Gradients.make_rdm1e(self, mo_energy[0], mo_coeff[0], mo_occ[0])
        dm1eb = krhf_grad.Gradients.make_rdm1e(self, mo_energy[1], mo_coeff[1], mo_occ[1])
        return cp.stack((dm1ea,dm1eb), axis=0)

    grad_elec = grad_elec
    extra_force = krhf_grad.Gradients.extra_force
    as_scanner = krhf_grad.Gradients.as_scanner
    _finalize = krhf_grad.Gradients._finalize
    kernel = krhf_grad.Gradients.kernel
