# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

# This function is only available in pyscf-2.8 or later
def extract_pgto_params(cell, op='diffused'):
    '''A helper function to extract exponents and contraction coefficients for
    estimate_xxx function
    '''
    es = []
    cs = []
    if op == 'diffused':
        precision = cell.precision
        for i in range(cell.nbas):
            e = cell.bas_exp(i)
            c = abs(cell._libcint_ctr_coeff(i)).max(axis=1)
            l = cell.bas_angular(i)
            # A quick estimation for the radius that each primitive GTO vanishes
            r2 = np.log(c**2 / precision * 10**l) / e
            idx = r2.argmax()
            es.append(e[idx])
            cs.append(c[idx].max())
    elif op == 'compact':
        precision = cell.precision
        for i in range(cell.nbas):
            e = cell.bas_exp(i)
            c = abs(cell._libcint_ctr_coeff(i)).max(axis=1)
            l = cell.bas_angular(i)
            # A quick estimation for the resolution of planewaves that each
            # primitive GTO requires
            ke = np.log(c**2 / precision * 50**l) * e
            idx = ke.argmax()
            es.append(e[idx])
            cs.append(c[idx].max())
    else:
        raise RuntimeError(f'Unsupported operation {op}')
    return np.array(es), np.array(cs)
