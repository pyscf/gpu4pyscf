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
NACV for TDDFT-RIS
'''

# This example will gives the derivative coupling (DC),
# also known as NACME (non-adiabatic coupling matrix element) 
# between ground and excited states.

import numpy as np
import pyscf
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
mf.conv_tol = 1e-10
mf.kernel()

td = mf.TDA().set(nstates=5) 
td.kernel() # [ 7.81949919  9.71029362 10.13398432 12.10163229 13.93675959]

nac = td.nac_method()
nac.states=(1,2) 
nac.kernel()
"""
--------- TDA nonadiabatic derivative coupling for states 1 and 2----------
         x                y                z
0 O    -0.0975602441    -0.0000000000    -0.0000000000
1 H     0.0548213338    -0.0000000000     0.0360881697
2 H     0.0548213338     0.0000000000    -0.0360881697
--------- TDA nonadiabatic derivative coupling for states 1 and 2 after E scaled (divided by E)----------
         x                y                z
0 O    -1.4040391809    -0.0000000000    -0.0000000000
1 H     0.7889617464    -0.0000000000     0.5193632370
2 H     0.7889617464     0.0000000000    -0.5193632370
--------- TDA nonadiabatic derivative coupling for states 1 and 2 with ETF----------
         x                y                z
0 O    -0.0965494688    -0.0000000000    -0.0000000000
1 H     0.0482746550    -0.0000000000     0.0378920920
2 H     0.0482746550     0.0000000000    -0.0378920920
--------- TDA nonadiabatic derivative coupling for states 1 and 2 with ETF after E scaled (divided by E)----------
         x                y                z
0 O    -1.3894925990    -0.0000000000    -0.0000000000
1 H     0.6947451570    -0.0000000000     0.5453244023
2 H     0.6947451570     0.0000000000    -0.5453244023
----------------------------------------------
"""

td_ris = ris.TDA(mf=mf.to_gpu(), nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
td_ris.conv_tol = 1.0E-6
td_ris.kernel() # [ 7.56352157  9.65590898 10.12364072 12.09873136 13.921372  ]
print(td_ris.energies.get())

nac_ris = td_ris.nac_method()
nac_ris.states=(1,2)
nac_ris.kernel()
"""
--------- TDA nonadiabatic derivative coupling for states 1 and 2----------
         x                y                z
0 O     0.1009731844     0.0000000000    -0.0000000014
1 H    -0.0575662857    -0.0000000000    -0.0380920213
2 H    -0.0575662879     0.0000000000     0.0380920227
--------- TDA nonadiabatic derivative coupling for states 1 and 2 after E scaled (divided by E)----------
         x                y                z
0 O     1.3131508452     0.0000000000    -0.0000000181
1 H    -0.7486464564    -0.0000000000    -0.4953846935
2 H    -0.7486464849     0.0000000000     0.4953847117
--------- TDA nonadiabatic derivative coupling for states 1 and 2 with ETF----------
         x                y                z
0 O     0.1006324741     0.0000000000    -0.0000000015
1 H    -0.0503161533    -0.0000000000    -0.0395091578
2 H    -0.0503161556     0.0000000000     0.0395091592
--------- TDA nonadiabatic derivative coupling for states 1 and 2 with ETF after E scaled (divided by E)----------
         x                y                z
0 O     1.3087199253     0.0000000000    -0.0000000189
1 H    -0.6543588731    -0.0000000000    -0.5138144769
2 H    -0.6543589035     0.0000000000     0.5138144958
----------------------------------------------
"""

print("defference for excitation energy between TDA and TDA-ris (in eV)")
print(td.e*27.21138602 - td_ris.energies.get())
print()
"""
[0.25597762 0.05438464 0.01034359 0.00290093 0.01538759]
"""
print("CIS derivative coupling without ETF")
print(np.abs(nac.de_scaled) - np.abs(nac_ris.de_scaled))
print("norm of difference", np.linalg.norm(np.abs(nac.de_scaled) - np.abs(nac_ris.de_scaled))) 
print()
"""
[[ 9.08883357e-02  7.20409145e-15 -1.80700507e-08]
 [ 4.03152900e-02  1.15468620e-14  2.39785435e-02]
 [ 4.03152615e-02  1.77621884e-16  2.39785253e-02]]
 0.11252232092869598
"""
print("difference for CIS derivative coupling with ETF")
print(np.abs(nac.de_etf_scaled) - np.abs(nac_ris.de_etf_scaled))
print("norm of difference", np.linalg.norm(np.abs(nac.de_etf_scaled) - np.abs(nac_ris.de_etf_scaled)))
print()
"""
[[ 8.07726737e-02  1.38040250e-14 -1.88986440e-08]
 [ 4.03862838e-02  1.32300610e-14  3.15099254e-02]
 [ 4.03862535e-02 -6.54111691e-16  3.15099065e-02]]
0.10849919731015174
"""

"""
Using the ris-approximated Z-vector solver rather than the standard Z-vector solver.
"""
nac_ris = td_ris.nac_method()
nac_ris.ris_zvector_solver = True # Use ris-approximated Z-vector solver
nac_ris.states=(1,2)
nac_ris.kernel()

