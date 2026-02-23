# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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


import numpy as np
import cupy
import pyscf
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.df.grad import rhf as df_rhf_grad

def get_veff(ks_grad, mol=None, dm=None, verbose=None):

    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    mf = ks_grad.base
    mf.with_df.reset() # Release GPU memory
    if dm is None: dm = mf.make_rdm1()

    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=False)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    if ks_grad.grid_response:
        log.debug('Compute XC deriviatives with grid response')
        exc, exc1 = rks_grad.get_exc_full_response(
                ni, mol, grids, mf.xc, dm, verbose=log)
        #log.debug1('sum(grids response) %s', exc.sum(axis=0))
        #log.debug1('grids response %s', exc)
        exc1 += exc/2
    else:
        exc, exc1 = rks_grad.get_exc(ni, mol, grids, mf.xc, dm, verbose=log)
    t0 = log.timer('vxc total', *t0)

    if mf.do_nlc():
        enlc1_per_atom, enlc1_grid = rks_grad._get_denlc(ks_grad, mol, dm)
        exc1 += enlc1_per_atom
        if ks_grad.grid_response:
            exc1 += enlc1_grid/2

    exc1 += ks_grad.jk_energy_per_atom(
        dm, j_factor=1, k_factor=hyb, hermi=1, verbose=log) * .5

    if ni.libxc.is_hybrid_xc(mf.xc) and omega != 0:  # For range separated Coulomb operator
        beta = alpha - hyb
        ek_lr = ks_grad.jk_energy_per_atom(
            dm, j_factor=0, k_factor=beta, hermi=1, omega=omega, verbose=log) * .5
        exc1 += ek_lr
    return exc1

class Gradients(rks_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    get_veff = get_veff
    jk_energy_per_atom = df_rhf_grad.Gradients.jk_energy_per_atom

Grad = Gradients
