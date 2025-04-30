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


# Example of TDDFT-ris


import pyscf
from gpu4pyscf.dft import rks
import gpu4pyscf.tdscf.ris as ris

atom ='''
C         -4.89126        3.29770        0.00029
H         -5.28213        3.05494       -1.01161
O         -3.49307        3.28429       -0.00328
H         -5.28213        2.58374        0.75736
H         -5.23998        4.31540        0.27138
H         -3.22959        2.35981       -0.24953
'''

mol = pyscf.M(atom=atom, basis='def2-svp', verbose=4)
mf = rks.RKS(mol, xc='wb97x').density_fit()

e_dft = mf.kernel()
print(f"total energy = {e_dft}")



''' TDDFT-ris'''
td = ris.TDDFT(mf=mf.to_gpu(), nstates=10, spectra=True)
td.kernel()
# energies, X, Y, oscillator_strength, rotatory_strength = td.kernel()

energies = td.energies
# X = td.X
# Y = td.Y
oscillator_strength = td.oscillator_strength
rotatory_strength = td.rotatory_strength

print("TDDFT-ris ex energies", energies)
print("TDDFT-ris oscillator_strength", oscillator_strength)

''' TDA-ris'''
td = ris.TDA(mf=mf.to_gpu(), nstates=10)
td.kernel()
# energies, X, oscillator_strength, rotatory_strength = td.kernel()

energies = td.energies
# X = td.X
oscillator_strength = td.oscillator_strength
rotatory_strength = td.rotatory_strength

print("TDA-ris ex energies", energies)
print("TDA-ris oscillator_strength", oscillator_strength)



