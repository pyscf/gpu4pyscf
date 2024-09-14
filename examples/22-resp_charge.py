# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
