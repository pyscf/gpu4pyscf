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

from pyscf import gto
from gpu4pyscf.dft.ucdft import CDFT_UKS
from gpu4pyscf import dft
import cupy as cp


mol = gto.Mole()
mol.atom = '''O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587'''
mol.basis = '6-31g'
mol.charge = 0
mol.spin = 0
mol.build()

# Define constraints (list of lists)
# Constrain O (atom 0) to 8.1 electrons
# Constrain first H (atom 1) to 0.95 electrons
charge_constraints = [ [0, 1], [8.1, 0.95] ]
    
mf = CDFT_UKS(mol, charge_constraints=charge_constraints)
mf.grids.atom_grid = (99, 590)
mf.xc = 'b3lyp'
mf.kernel() #  -76.2906028922253

print("\n>>> Analysis of Results")
print(f"Converged Lagrange Multipliers V: {mf.v_lagrange}") # [ 0.32179487 -0.04154594]

# Verification of constraints
dm = mf.make_rdm1()
projs = mf.build_projectors()
O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
H1_charge = cp.trace(dm[0] @ projs[1]) + cp.trace(dm[1] @ projs[1])
print(f"Number of projs: {len(projs)}")
print(f"Target O:  {charge_constraints[1][0]}")
print(f"Result O:  {float(O_charge):.6f}") # 8.100000
print(f"Target H1:  {charge_constraints[1][1]}")
print(f"Result H1:  {float(H1_charge):.6f}") # 0.950000

# Get the MO energies with physical meanings
# mf.mo_energy is calculated from Fock + V_cons
# mf.get_canonical_mo() will return the MO energies calculated from Fock
mo_energy = mf.get_canonical_mo()[0]
print(mo_energy)
"""
[[-19.36209987  -1.12444181  -0.60693476  -0.46851442  -0.41940472
    0.03791074   0.12346241   0.79344581   0.79368322   0.80307202
    0.91837328   0.9845149    1.35833387]
 [-19.36209987  -1.1244418   -0.60693476  -0.46851441  -0.41940472
    0.03791073   0.1234624    0.79344581   0.79368322   0.80307202
    0.91837328   0.9845149    1.35833387]]
"""


"""
If the default nested iteration method not converge, 
you can try to use the Newton-Raphson method.
"""
mf_soscf = mf.newton()
# Newton-Raphson method
# This will converge in 1 iteration.
mf_soscf.kernel() # -76.2906028922247


"""
We provide another method to solve the constrained DFT problem, 
which is the penalty method. If the above methods not converge, 
you can try the penalty method.
"""
mf_penalty = CDFT_UKS(mol, charge_constraints=charge_constraints,
              method='penalty',
              penalty_weight=10.0)
mf_penalty.grids.atom_grid = (99, 590)
mf_penalty.xc = 'b3lyp'
# penalty method converges much easily using soscf.
mf_penalty = mf_penalty.newton_penalty()
mf_penalty.kernel() # -76.2955818883445


"""
Penalty method will always have errors.
"""
dm = mf_penalty.make_rdm1()
projs = mf_penalty._scf.build_projectors()
O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
H1_charge = cp.trace(dm[0] @ projs[1]) + cp.trace(dm[1] @ projs[1])
print(f"Number of projs: {len(projs)}")
print(f"Target O:  {charge_constraints[1][0]}")
print(f"Result O:  {float(O_charge):.6f}") # 8.115437
print(f"Target H1:  {charge_constraints[1][1]}")
print(f"Result H1:  {float(H1_charge):.6f}") # 0.947531


"""
Analytical gradient is only supported for MINAO partition method (projected mulliken).
And the penalty method is not supported.
"""
g = mf.Gradients()
g.kernel()
"""
--------------- CDFT_UKS gradients ---------------
         x                y                z
0 O     0.0000000000    -0.0150321699     0.1229336008
1 H    -0.0000000000     0.0811590504    -0.0672953529
2 H     0.0000000000    -0.0661268790    -0.0556389656
----------------------------------------------
"""
g = mf_soscf.Gradients()
g.kernel()
"""
--------------- SecondOrderCDFT_UKS gradients ---------------
         x                y                z
0 O     0.0000000000    -0.0150321704     0.1229335989
1 H    -0.0000000000     0.0811590496    -0.0672953523
2 H     0.0000000000    -0.0661268777    -0.0556389644
----------------------------------------------
"""