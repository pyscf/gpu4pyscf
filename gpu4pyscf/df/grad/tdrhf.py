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


from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.df.grad import rhf as rhf_grad_df
from gpu4pyscf.tdscf import rhf as tdrhf
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf import __config__
from gpu4pyscf.lib import logger
from functools import reduce
import cupy as cp


def get_veff(td_grad, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
    dm_scf=False
    if omega != 0.0:
        vj, vk, vjaux, vkaux = td_grad.get_jk(mol, dm, omega=omega, hermi=hermi, dm_scf=dm_scf)
    else:
        vj, vk, vjaux, vkaux = td_grad.get_jk(mol, dm, hermi=hermi, dm_scf=dm_scf)
    vhf = vj * j_factor - vk * .5 * k_factor
    if td_grad.auxbasis_response:
        e1_aux = vjaux * j_factor - vkaux * .5 * k_factor
        logger.debug1(td_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
    else:
        e1_aux = None
    vhf = tag_array(vhf, aux=e1_aux)
    
    return vhf


class Gradients(tdrhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}
    def __init__(self, td):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        tdrhf_grad.Gradients.__init__(self, td)

    auxbasis_response = True
    get_jk = rhf_grad_df.get_jk

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)

    get_veff = get_veff

Grad = Gradients
