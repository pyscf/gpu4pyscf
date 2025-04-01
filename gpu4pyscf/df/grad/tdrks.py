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


from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.df.grad import tdrhf as tdrhf_grad_df
from gpu4pyscf.tdscf import rks as tdrks
from gpu4pyscf.grad import tdrks as tdrks_grad
from gpu4pyscf import __config__

class Gradients(tdrks_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}
    def __init__(self, td):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        tdrks_grad.Gradients.__init__(self, td)

    auxbasis_response = True
    get_jk = tdrhf_grad_df.get_jk

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrks.TDDFT) or isinstance(self.base, tdrks.TDA)

    get_veff = tdrhf_grad_df.get_veff

Grad = Gradients
