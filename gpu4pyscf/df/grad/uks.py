# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import cupy
from pyscf import lib
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad.uhf import get_jk
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import logger


def get_veff(ks_grad, mol=None, dm=None, verbose=None):
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
        exc, exc1 = uks_grad.get_exc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            raise NotImplementedError
    else:
        exc, exc1 = uks_grad.get_exc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, exc1_nlc = uks_grad.get_nlc_exc(
                ni, mol, nlcgrids, xc, dm, mf.mo_coeff, mf.mo_occ,
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc1 += exc1_nlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    aoslices = mol.aoslice_by_atom()
    exc1 = [exc1[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    exc1 = cupy.asarray(exc1)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        mo_a, mo_b = ks_grad.base.mo_coeff
        mo_occa, mo_occb = ks_grad.base.mo_occ
        ej0, ejaux0 = ks_grad.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa)
        ej1, ejaux1 = ks_grad.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb)
        ej0_m1, ejaux0_m1 = ks_grad.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa, dm2=dm[1])
        ej1_m0, ejaux1_m0 = ks_grad.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb, dm2=dm[0])
        if ks_grad.auxbasis_response:
            e1_aux = ejaux0 + ejaux1 + ejaux0_m1 + ejaux1_m0
        exc1 += ej0 + ej1 + ej0_m1 + ej1_m0
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        mo_a, mo_b = ks_grad.base.mo_coeff
        mo_occa, mo_occb = ks_grad.base.mo_occ
        ej0, ek0, ejaux0, ekaux0 = ks_grad.get_jk(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa)
        ej1, ek1, ejaux1, ekaux1 = ks_grad.get_jk(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb)
        ej0_m1, ejaux0_m1 = ks_grad.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa, dm2=dm[1])
        ej1_m0, ejaux1_m0 = ks_grad.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb, dm2=dm[0])
        ej = ej0 + ej1 + ej0_m1 + ej1_m0
        ek = (ek0 + ek1) * hyb
        if ks_grad.auxbasis_response:
            ej_aux = ejaux0 + ejaux1 + ejaux0_m1 + ejaux1_m0
            ek_aux = (ekaux0+ekaux1) * hyb

        if omega != 0:
            mocc0 = ks_grad.base.mo_occ[0]
            mocc1 = ks_grad.base.mo_occ[1]
            mo_coeff0 = ks_grad.base.mo_coeff[0]
            mo_coeff1 = ks_grad.base.mo_coeff[1]
            ek_lr0, ekaux_lr0 = ks_grad.get_k(mol, dm[0], mo_coeff=mo_coeff0, mo_occ=mocc0, omega=omega)
            ek_lr1, ekaux_lr1 = ks_grad.get_k(mol, dm[1], mo_coeff=mo_coeff1, mo_occ=mocc1, omega=omega)
            ek += (ek_lr0 + ek_lr1) * (alpha-hyb)
            if ks_grad.auxbasis_response:
                ek_aux += (ekaux_lr0 + ekaux_lr1) * (alpha-hyb)

        exc1 += ej - ek
        if ks_grad.auxbasis_response:
            e1_aux = ej_aux - ek_aux

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
    else:
        e1_aux = None

    exc1 = tag_array(exc1, aux=e1_aux, exc1_grid=exc)

    return exc1


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
