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

from gpu4pyscf.lib import logger
from gpu4pyscf.df.grad import uks as uks_grad
from gpu4pyscf.grad.ucdft import get_constraint_force


class Gradients(uks_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        uks_grad.Gradients.__init__(self, mf)

    auxbasis_response = True

    def get_veff(self, mol=None, dm=None, verbose=None):
        """
        Calculate the gradient response from the constraint potential.
        """
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        
        logger.info(self.base, "Calculating constraint gradient contributions (Minao)...")
        # Note: Do not add force here. Veff is doubled in get_elec.
        self._dE_constraint = get_constraint_force(self, dm)
        return super().get_veff(mol, dm, verbose)

    def extra_force(self, atom_id, envs):
        force = super().extra_force(atom_id, envs)
        
        if self._dE_constraint is not None:
            force += self._dE_constraint[atom_id]
            
        return force

Grad = Gradients
