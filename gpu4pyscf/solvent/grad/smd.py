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
Gradient of SMD solvent model
'''
# pylint: disable=C0103

import numpy as np
import cupy
#from cupyx import scipy, jit
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from gpu4pyscf.solvent import pcm, smd
from gpu4pyscf.solvent.grad import pcm as pcm_grad
from gpu4pyscf.lib import logger
from gpu4pyscf.grad.rhf import GradientsBase

def get_cds(smdobj):
    return smd.get_cds_legacy(smdobj)[1]

grad_solver = pcm_grad.grad_solver

def make_grad_object(base_method):
    '''Create nuclear gradients object with solvent contributions for the given
    solvent-attached method based on its gradients method in vaccum
    '''
    if isinstance(base_method, GradientsBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a gradient object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    # create the Gradients in vacuum. Cannot call super().Gradients() here
    # because other dynamic corrections might be applied to the base_method. The
    # super() class might discard these corrections .
    vac_grad = base_method.undo_solvent().Gradients()
    # The base method for vac_grad discards the with_solvent. Change its base to
    # the solvent-attached base method
    vac_grad.base = base_method
    name = with_solvent.__class__.__name__ + vac_grad.__class__.__name__
    return lib.set_class(WithSolventGrad(vac_grad),
                         (WithSolventGrad, vac_grad.__class__), name)

class WithSolventGrad:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute', 'de_cds'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.grad import smd      # type: ignore
        grad_method = self.undo_solvent().to_cpu()
        return smd.make_grad_object(grad_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        logger.debug(self, 'Compute gradients from solvents')
        self.de_solvent = self.base.with_solvent.grad(dm)
        logger.debug(self, 'Compute gradients from solutes')
        self.de_solute  = super().kernel(*args, **kwargs)
        self.de_cds     = get_cds(self.base.with_solvent)
        self.de = self.de_solute + self.de_solvent + self.de_cds
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


