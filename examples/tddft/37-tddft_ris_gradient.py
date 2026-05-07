#!/usr/bin/env python
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

'''
Gradient for TDDFT-RIS
'''

import pyscf
import numpy as np
import gpu4pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.tdscf import ris

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2tzvp')

mf = rks.RKS(mol, xc='pbe0') # -76.3773133945678
mf.kernel()

td = mf.TDA()
td.nstates=5
td.kernel() # [ 7.81949919  9.71029362 10.13398432 12.10163229 13.93675959] (eV)

g = td.nuc_grad_method()
g.state=1
g.kernel()
"""
--------- TDA gradients for state 1 ----------
         x                y                z
0 O     0.0000000000     0.0000000000    -0.0949023769
1 H     0.0627472634    -0.0000000000     0.0474538726
2 H    -0.0627472634    -0.0000000000     0.0474538726
----------------------------------------------
"""

td_ris = ris.TDA(mf=mf.to_gpu(), nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
td_ris.conv_tol = 1.0E-4
td_ris.kernel() # [ 7.56352157  9.65590899 10.1236409  12.09873137 13.921372  ] (eV)

g_ris = td_ris.nuc_grad_method()
g_ris.state=1
g_ris.kernel()
"""
--------- TDA gradients for state 1 ----------
         x                y                z
0 O     0.0000000106    -0.0000000000    -0.0969465222
1 H     0.0674194961     0.0000000000     0.0484759423
2 H    -0.0674195066     0.0000000000     0.0484759501
----------------------------------------------
"""

print("defference for excitation energy between TDA and TDA-ris (in eV)")
print(td.e*27.21138602 - td_ris.energies.get())
print()
"""
[0.25597762 0.05438464 0.01034341 0.00290092 0.01538759]
"""
print("defference for gradient between TDA and TDA-ris (in Hartree/Bohr)")
print(g.de - g_ris.de)
"""
[[-1.05589088e-08 -1.97977506e-15  2.04414538e-03]
 [-4.67223270e-03  9.83637092e-16 -1.02206969e-03]
 [ 4.67224325e-03  1.00292678e-15 -1.02207751e-03]]
"""
print("norm of the diff")
print(np.linalg.norm(g.de - g_ris.de))
"""
0.007065933384199997
"""

"""
Using the ris-approximated Z-vector solver rather than the standard Z-vector solver.
"""
g_ris = td_ris.nuc_grad_method()
g_ris.ris_zvector_solver = True # Use ris-approximated Z-vector solver
g_ris.state=1
g_ris.kernel()