# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import cupy
from pyscf import lib
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad.uhf import get_jk
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import logger


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
    if mf.do_nlc():
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
        exc, vxc_tmp = uks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            raise NotImplementedError
    else:
        exc, vxc_tmp = uks_grad.get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = uks_grad.get_nlc_vxc(
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
    vxc = contract('nik,ik->ni', tmp, occ_coeff0)
    tmp = contract('nij,jk->nik', vxc_tmp[1], occ_coeff1)
    vxc+= contract('nik,ik->ni', tmp, occ_coeff1)

    aoslices = mol.aoslice_by_atom()
    vxc = [vxc[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    vxc = cupy.asarray(vxc)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        mo_a, mo_b = ks_grad.base.mo_coeff
        mo_occa, mo_occb = ks_grad.base.mo_occ
        vj0, vjaux0 = ks_grad.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa)
        vj1, vjaux1 = ks_grad.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb)
        vj0_m1, vjaux0_m1 = ks_grad.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa, dm2=dm[1])
        vj1_m0, vjaux1_m0 = ks_grad.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb, dm2=dm[0])
        if ks_grad.auxbasis_response:
            e1_aux = vjaux0 + vjaux1 + vjaux0_m1 + vjaux1_m0
        vxc += vj0 + vj1 + vj0_m1 + vj1_m0
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        mo_a, mo_b = ks_grad.base.mo_coeff
        mo_occa, mo_occb = ks_grad.base.mo_occ
        vj0, vk0, vjaux0, vkaux0 = ks_grad.get_jk(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa)
        vj1, vk1, vjaux1, vkaux1 = ks_grad.get_jk(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb)
        vj0_m1, vjaux0_m1 = ks_grad.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa, dm2=dm[1])
        vj1_m0, vjaux1_m0 = ks_grad.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb, dm2=dm[0])
        vj = vj0 + vj1 + vj0_m1 + vj1_m0
        vk = (vk0 + vk1) * hyb
        if ks_grad.auxbasis_response:
            vj_aux = vjaux0 + vjaux1 + vjaux0_m1 + vjaux1_m0
            vk_aux = (vkaux0+vkaux1) * hyb

        if omega != 0:
            vk_lr0, vkaux_lr0 = ks_grad.get_k(mol, dm[0], mo_coeff=ks_grad.base.mo_coeff[0], mo_occ=ks_grad.base.mo_occ[0], omega=omega)
            vk_lr1, vkaux_lr1 = ks_grad.get_k(mol, dm[1], mo_coeff=ks_grad.base.mo_coeff[1], mo_occ=ks_grad.base.mo_occ[1], omega=omega)
            vk += (vk_lr0 + vk_lr1) * (alpha-hyb)
            if ks_grad.auxbasis_response:
                vk_aux += (vkaux_lr0 + vkaux_lr1) * (alpha-hyb)

        vxc += vj - vk
        if ks_grad.auxbasis_response:
            e1_aux = vj_aux - vk_aux

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
    else:
        e1_aux = None

    vxc = tag_array(vxc, aux=e1_aux)

    return vxc


class Gradients(uks_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        uks_grad.Gradients.__init__(self, mf)

    # Whether to include the response of DF auxiliary basis when computing
    # nuclear gradients of J/K matrices
    auxbasis_response = True
    get_jk = get_jk

    def get_j(self, mol=None, dm=None, hermi=0, mo_coeff=None, mo_occ=None, dm2 = None, omega=None):
        vj, _, vjaux, _ = self.get_jk(mol, dm, with_k=False, mo_coeff=mo_coeff, mo_occ=mo_occ, dm2=dm2, omega=omega)
        return vj, vjaux

    def get_k(self, mol=None, dm=None, hermi=0, mo_coeff=None, mo_occ=None, dm2 = None, omega=None):
        _, vk, _, vkaux = self.get_jk(mol, dm, with_j=False, mo_coeff=mo_coeff, mo_occ=mo_occ, dm2=dm2, omega=omega)
        return vk, vkaux

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        e1 = uks_grad.Gradients.extra_force(self, atom_id, envs)
        if self.auxbasis_response:
            e1 += envs['dvhf'].aux[atom_id]
        return e1

Grad = Gradients
