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
import pyscf
from pyscf import lib
from pyscf.df.grad import rks as df_rks_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import logger

def get_veff(ks_grad, mol=None, dm=None, verbose=None):

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

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, exc1 = rks_grad.get_exc_full_response(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
        #logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, exc1 = rks_grad.get_exc(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
    t0 = logger.timer(ks_grad, 'vxc total', *t0)

    aoslices = mol.aoslice_by_atom()
    exc1 = [exc1[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
    exc1 = cupy.asarray(exc1)

    if mf.do_nlc():
        enlc1_per_atom, enlc1_grid = rks_grad._get_denlc(ks_grad, mol, dm, max_memory)
        exc1 += enlc1_per_atom
        if ks_grad.grid_response:
            exc += enlc1_grid

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        ej, ejaux = ks_grad.get_j(mol, dm)
        exc1 += ej
        if ks_grad.auxbasis_response:
            e1_aux = ejaux
    else:
        ej, ek, ejaux, ekaux = ks_grad.get_jk(mol, dm)

        if ks_grad.auxbasis_response:
            ek_aux = ekaux * hyb
        ek *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            ek_lr, ekaux_lr = ks_grad.get_k(mol, dm, omega=omega)
            ek += ek_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                ek_aux += ekaux_lr * (alpha - hyb)

        exc1 += ej - ek * .5
        if ks_grad.auxbasis_response:
            e1_aux = ejaux - ek_aux * .5

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
    else:
        e1_aux = None
    exc1 = tag_array(exc1, aux=e1_aux, exc1_grid=exc)
    return exc1

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
        e1 = rks_grad.Gradients.extra_force(self, atom_id, envs)
        if self.auxbasis_response:
            e1 += envs['dvhf'].aux[atom_id]
        return e1

Grad = Gradients
