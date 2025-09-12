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

from pyscf import lib
from gpu4pyscf.dft import numint
from pyscf import __config__

class NumInt2C(lib.StreamObject, numint.LibXCMixin):
    '''Numerical integration methods for 2-component basis (used by GKS)'''

    # collinear schemes:
    #   'col' (collinear, by default)
    #   'ncol' (non-collinear)
    #   'mcol' (multi-collinear)
    collinear = getattr(__config__, 'dft_numint_RnumInt_collinear', 'col')
    spin_samples = getattr(__config__, 'dft_numint_RnumInt_spin_samples', 770)
    collinear_thrd = getattr(__config__, 'dft_numint_RnumInt_collinear_thrd', 0.99)
    collinear_samples = getattr(__config__, 'dft_numint_RnumInt_collinear_samples', 200)

    make_mask = staticmethod(numint.make_mask)
    eval_ao = staticmethod(numint.eval_ao)
    eval_rho = staticmethod(eval_rho)

    eval_rho1 = NotImplemented
    eval_rho2 = NotImplemented
    cache_xc_kernel = NotImplemented
    cache_xc_kernel1 = NotImplemented
    get_rho = NotImplemented
    _gks_mcol_vxc = NotImplemented
    _gks_mcol_fxc = NotImplemented
    nr_vxc = NotImplemented
    nr_nlc_vxc = NotImplemented
    nr_fxc = NotImplemented
    get_fxc = nr_gks_fxc = nr_fxc

    eval_xc_eff = NotImplemented
    mcfun_eval_xc_adapter = NotImplemented

    block_loop = NotImplemented
    _gen_rho_evaluator = NotImplemented

    def _to_numint1c(self):
        '''Converts to the associated class to handle collinear systems'''
        return self.view(numint.NumInt)
