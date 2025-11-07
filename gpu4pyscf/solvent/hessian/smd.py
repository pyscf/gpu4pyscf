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

'''
Hessian SMD solvent model
'''
# pylint: disable=C0103

import numpy as np
from pyscf import lib
from gpu4pyscf import scf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.solvent.grad import smd as smd_grad
from gpu4pyscf.solvent.hessian import pcm as pcm_hess
from gpu4pyscf.hessian.rhf import HessianBase, _ao2mo

def get_cds(smdobj):
    mol = smdobj.mol.copy()
    smdobj_tmp = smdobj.copy()
    def smd_grad_scanner(mol):
        smdobj_tmp.reset(mol)
        return smd_grad.get_cds(smdobj_tmp)

    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()

    coords = mol.atom_coords(unit='B')
    coords_backup = coords.copy()
    eps = 1e-4
    natm = mol.natm
    hess_cds = np.zeros([natm,natm,3,3])
    for ia in range(mol.natm):
        for j in range(3):
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            grad0_cds = smd_grad_scanner(mol)

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            grad1_cds = smd_grad_scanner(mol)
            hess_cds[ia,:,j] = (grad0_cds - grad1_cds) / (2.0 * eps)
            coords[ia,j] = coords_backup[ia,j]
    t1 = log.timer_debug1('solvent energy', *t1)
    return hess_cds # hartree

def make_hess_object(base_method):
    '''Create nuclear hessian object with solvent contributions for the given
    solvent-attached method based on its hessian method in vaccum
    '''
    if isinstance(base_method, HessianBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a hessian object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for Hessian')

    vac_hess = base_method.undo_solvent().Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithSolventHess(vac_hess),
                         (WithSolventHess, vac_hess.__class__), name)

class WithSolventHess:

    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {'de_solvent', 'de_solute', 'de_cds'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        hess_method = self.base.to_cpu().Hessian()
        return utils.to_cpu(self, hess_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        if self.base.with_solvent.frozen_dm0_for_finite_difference_without_response is not None:
            raise NotImplementedError("frozen_dm0_for_finite_difference_without_response not implemented for PCM Hessian")

        with lib.temporary_env(self.base.with_solvent, equilibrium_solvation=True):
            logger.debug(self, 'Compute hessian from solutes')
            self.de_solute = super().kernel(*args, **kwargs)
        logger.debug(self, 'Compute hessian from solvents')
        self.de_solvent = self.base.with_solvent.hess(dm)
        self.de_cds = get_cds(self.base.with_solvent)
        self.de = self.de_solute + self.de_solvent + self.de_cds
        return self.de

    make_h1 = pcm_hess.WithSolventHess.make_h1

    get_veff_resp_mo = pcm_hess.WithSolventHess.get_veff_resp_mo

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
