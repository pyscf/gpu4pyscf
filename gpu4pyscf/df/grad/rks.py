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
import pyscf
from pyscf import lib
from pyscf.df.grad import rks as df_rks_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import logger

def get_veff(ks_grad, mol=None, dm=None):

    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = logger.init_timer(ks_grad)

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(with_non0tab=False)

    nlcgrids = None
    if mf.do_nlc():
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(with_non0tab=False)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = rks_grad.get_vxc_full_response(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
        #logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        if mf.do_nlc():
            raise NotImplementedError
    else:
        exc, vxc = rks_grad.get_vxc(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc total', *t0)

    # this can be moved into vxc calculations
    occ_coeff = cupy.asarray(mf.mo_coeff[:, mf.mo_occ>0.5], order='C')
    tmp = contract('nij,jk->nik', vxc, occ_coeff)
    vxc = 2.0*contract('nik,ik->ni', tmp, occ_coeff)

    aoslices = mol.aoslice_by_atom()
    vxc = [vxc[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    vxc = cupy.asarray(vxc)
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vj, vjaux = ks_grad.get_j(mol, dm)
        vxc += vj
        if ks_grad.auxbasis_response:
            e1_aux = vjaux
    else:
        vj, vk, vjaux, vkaux = ks_grad.get_jk(mol, dm)

        if ks_grad.auxbasis_response:
            vk_aux = vkaux * hyb
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            vk_lr, vkaux_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk_aux += vkaux_lr * (alpha - hyb)

        vxc += vj - vk * .5
        if ks_grad.auxbasis_response:
            e1_aux = vjaux - vk_aux * .5

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
    else:
        e1_aux = None
    vxc = tag_array(vxc, aux=e1_aux)
    return vxc

class Gradients(rks_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        rks_grad.Gradients.__init__(self, mf)

    # Whether to include the response of DF auxiliary basis when computing
    # nuclear gradients of J/K matrices
    auxbasis_response = True

    get_jk = df_rhf_grad.Gradients.get_jk
    grad_elec = df_rhf_grad.Gradients.grad_elec
    get_veff = get_veff

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        vj, _, vjaux, _ = self.get_jk(mol, dm, with_k=False, omega=omega)
        return vj, vjaux

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        _, vk, _, vkaux = self.get_jk(mol, dm, with_j=False, omega=omega)
        return vk, vkaux

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['dvhf'].aux[atom_id]
        else:
            return 0

Grad = Gradients