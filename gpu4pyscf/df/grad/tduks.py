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


from gpu4pyscf.df import df
from gpu4pyscf.df.grad import tduhf as tduhf_grad_df
from gpu4pyscf.tdscf import uks as tduks
from gpu4pyscf.grad import tduks as tduks_grad

class Gradients(tduks_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tduks.TDDFT) or isinstance(self.base, tduks.TDA)

    get_veff = tduhf_grad_df.Gradients.get_veff

Grad = Gradients
