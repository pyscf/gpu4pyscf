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
Nonadiabatic coupling vectors between ground and excited states for RHF
'''

# This example will gives the derivative coupling (DC),
# also known as NACME (non-adiabatic coupling matrix element) 
# between ground and excited states.

import pyscf
import gpu4pyscf
from gpu4pyscf.scf import hf

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='ccpvdz')

mf = hf.RHF(mol) #  -76.0267656731119
mf.kernel()

td = mf.TDA().set(nstates=5) # TDHF is OK
td.kernel() # [ 9.21540892 10.99036172 11.83380819 13.62301694 15.06349085]

nac = gpu4pyscf.nac.tdrhf.NAC(td)
nac.state=(0,1) # same as (1,0) 0 means ground state, 1 means the first excited state
nac.kernel()
'''
--------- TDA nonadiabatic derivative coupling for state 0 and 1----------
         x                y                z
0 O    -0.0000000000     0.0225763887     0.0000000000
1 H     0.0000000000     0.0321451453    -0.0000000000
2 H    -0.0000000000     0.0321451453    -0.0000000000
--------- TDA nonadiabatic derivative coupling for state 0 and 1 after E scaled (divided by E)----------
         x                y                z
0 O    -0.0000000000     0.0666638707     0.0000000000
1 H     0.0000000000     0.0949186265    -0.0000000000
2 H    -0.0000000000     0.0949186265    -0.0000000000
--------- TDA nonadiabatic derivative coupling for state 0 and 1 with ETF----------
         x                y                z
0 O    -0.0000000000    -0.1316160824     0.0000000000
1 H     0.0000000000     0.0658080412    -0.0000000000
2 H    -0.0000000000     0.0658080412    -0.0000000000
--------- TDA nonadiabatic derivative coupling for state 0 and 1 with ETF after E scaled (divided by E)----------
         x                y                z
0 O    -0.0000000000    -0.3886377757     0.0000000000
1 H     0.0000000000     0.1943188879    -0.0000000000
2 H    -0.0000000000     0.1943188879    -0.0000000000
----------------------------------------------
'''

print('-----------------------------------------------------')
print("Non-adiabatic coupling matrix element (NACME) between ground and first excited state")
print(nac.de)
print('-----------------------------------------------------')
print("NACME between ground and first excited state scaled by E (/E_ex)")
print(nac.de_scaled)
print('-----------------------------------------------------')
print("NACME between ground and first excited state with ETF (electron translation factor)")
# Without including the contribution of the electron translation factor (ETF), for some molecules, 
# the non-adiabatic coupling matrix element (NACME) may lack translational invariance, 
# which can further lead to errors in subsequent calculations such as MD simulations. 
# In this case, it is necessary to use the NACME that takes the ETF into account.
print(nac.de_etf)
print('-----------------------------------------------------')
print("NACME between ground and first excited state with ETF (electron translation factor) scaled by E (/E_ex)")
print(nac.de_etf_scaled)