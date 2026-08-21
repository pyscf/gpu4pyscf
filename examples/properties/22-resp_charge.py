#!/usr/bin/env python
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

###########################################################
#  Example of calculating two-stage RESP charge
###########################################################

import pyscf
from pyscf import scf
from gpu4pyscf.pop import esp

atom = """ C   1.45051389  -0.06628932   0.00000000
    H   1.75521613  -0.62865986  -0.87500146
    H   1.75521613  -0.62865986   0.87500146
    H   1.92173244   0.90485897   0.00000000
    C  -0.04233122   0.09849378   0.00000000
    O  -0.67064817  -1.07620915   0.00000000
    H  -1.60837259  -0.91016601   0.00000000
    O  -0.62675864   1.13160510   0.00000000"""
mol = pyscf.M(atom=atom, basis='6-31gs')
mol.cart = True    # PySCF uses spherical basis by default

mf = scf.RHF(mol)
mf.kernel()
dm = mf.make_rdm1()

# ESP charge
q0 = esp.esp_solve(mol, dm)
print('Fitted ESP charge')
print(q0)

# RESP charge // first stage fitting
q1 = esp.resp_solve(mol, dm)

# Add constraint: fix those charges in the second stage
# q2[4] = q1[4]
# q2[5] = q1[5]
# q2[6] = q1[6]
# q2[7] = q1[7]
sum_constraints = []
for i in range(4,8):
    sum_constraints.append([q1[i], [i]])

# Add constraints: same charges of hydrogens connected to first Carbon
# q2[1] = q2[2] = q2[3]
equal_constraints = [[1,2,3]]

# RESP charge // second stage fitting
q2 = esp.resp_solve(mol, dm, resp_a=1e-3,
                    sum_constraints=sum_constraints,
                    equal_constraints=equal_constraints)
print('Fitted RESP charge')
print(q2)
