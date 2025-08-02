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

from gpu4pyscf.nac import tdrhf as tdrhf_nac
from gpu4pyscf.df.grad.tdrhf import get_jk, get_veff
from gpu4pyscf.tdscf import rhf as tdrhf
from gpu4pyscf.lib import logger
from gpu4pyscf.df import df
from gpu4pyscf.lib.cupy_helper import tag_array


class NAC(tdrhf_nac.NAC):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}
    def __init__(self, td):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        tdrhf_nac.NAC.__init__(self, td)

    auxbasis_response = True
    get_jk = get_jk
    get_veff = get_veff

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)
