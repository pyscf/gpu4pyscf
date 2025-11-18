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
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2
from gpu4pyscf.pbc.gto import int1e

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
    s1 = mf_grad.get_ovlp(cell, kpts)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    # derivatives of the Veff contribution
    dvhf = mf_grad.get_veff(dm0, kpts) * 2
    t1 = log.timer('gradients of 2e part', *t0)

    dm0_sf = dm0[0] + dm0[1]
    ni = getattr(mf, "_numint", None)
    if isinstance(ni, multigrid.MultiGridNumInt):
        raise NotImplementedError(
            "Gradient with kpts not implemented with multigrid.MultiGridNumInt. "
            "Please use the default KNumInt or multigrid_v2.MultiGridNumInt instead.")
    elif isinstance(ni, multigrid_v2.MultiGridNumInt):
        # Attention: The orbital derivative of vpploc term is in multigrid_v2.get_veff_ip1() function.
        rho_g = multigrid_v2.evaluate_density_on_g_mesh(ni, dm0_sf, kpts)
        rho_g = rho_g[0,0]
        if cell._pseudo:
            dh1e = multigrid.eval_vpplocG_SI_gradient(cell, ni.mesh, rho_g) * nkpts
        else:
            dh1e = multigrid.eval_nucG_SI_gradient(cell, ni.mesh, rho_g) * nkpts

        dm_dmH = dm0_sf + dm0_sf.transpose(0,2,1).conj()
        dh1e_kin = int1e.int1e_ipkin(cell, kpts)
        aoslices = cell.aoslice_by_atom()
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            dh1e[ia] -= cp.einsum('kxij,kji->x', dh1e_kin[:,:,p0:p1,:], dm_dmH[:,:,p0:p1]).real
    else:
        hcore_deriv = mf_grad.hcore_generator(cell, kpts)
        dh1e = cp.empty([natm, 3])
        for ia in range(natm):
            h1ao = hcore_deriv(ia)
            dh1e[ia] = cp.einsum('kxij,kji->x', h1ao, dm0_sf).real

    if cell._pseudo:
        dm0_sf_cpu = dm0_sf.get()
        dh1e_pp_nonlocal = vppnl_nuc_grad(cell, dm0_sf_cpu, kpts = kpts)
        dh1e += cp.asarray(dh1e_pp_nonlocal)

    log.timer('gradients of 1e part', *t1)

    extra_force = np.empty([natm, 3])
    for ia in range(natm):
        extra_force[ia] = ensure_numpy(mf_grad.extra_force(ia, locals()))

    # nabla is applied on bra in vhf. *2 for the contributions of nabla|ket>
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dme0_sf = dme0[0] + dme0[1]
    aoslices = cell.aoslice_by_atom()
    ds = contract('kxij,kji->xi', s1, dme0_sf).real
    ds = (-2 * ds).get()
    ds = np.array([ds[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])
    de = (dh1e.get() + ds) / nkpts + dvhf + extra_force

    if log.verbose > logger.DEBUG:
        log.debug('gradients of electronic part')
        mf_grad._write(cell, de, atmlst)
    return de

class Gradients(krhf_grad.GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    def get_veff(self, dm, kpts):
        '''
        The energy contribution from the effective potential

        einsum('skxij,skji->x', veff, dm) / nkpts
        '''
        if self.base.rsjk is not None:
            from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
            with_rsjk = self.base.rsjk
            assert isinstance(with_rsjk, PBCJKMatrixOpt)
            if with_rsjk.supmol is None:
                with_rsjk.build()
            ejk = with_rsjk._get_ejk_sr_ip1(dm, kpts, exxdiv=self.base.exxdiv)
            ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts, exxdiv=self.base.exxdiv)
        else:
            ej = self.get_j(dm[0]+dm[1], kpts)
            ek = self.get_k(dm, kpts)
            ejk = ej - ek
        return ejk

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

    def get_stress(self):
        from gpu4pyscf.pbc.grad import kuhf_stress
        return kuhf_stress.kernel(self)
