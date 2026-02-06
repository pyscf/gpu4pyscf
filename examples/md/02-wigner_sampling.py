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
This example demonstrates  the wigner sampling method to generate the initial 
'''

import numpy as np
import pyscf
from pyscf.hessian import thermo
from gpu4pyscf.md.wigner_sampling import wigner_samples

mol = pyscf.M(
    atom='''
C   -1.302   0.206   0.000  
H   -0.768  -0.722   0.000  
H   -2.372   0.206   0.000  
C   -0.626   1.381   0.000  
H   -1.159   2.309   0.000  
H    0.444   1.381   0.000  
''',
    basis='sto-3g')
mf = mol.RHF()
mol_eq = mf.Gradients().optimizer().kernel()
print("Equilibrium coordinates (Angstrom):")
print(mol_eq.atom_coords(unit='a'))

mf_eq = mol_eq.RHF().run()
h = mf_eq.Hessian().kernel()
thermo_data = thermo.harmonic_analysis(mol_eq, h)

# Generate 50 initial conditions at 300 Kelvin
temperature = 300
samples = 50
xyz = mol_eq.atom_coords(unit='Bohr')
inits = wigner_samples(temperature, thermo_data['freq_wavenumber'], xyz,
                       thermo_data['norm_mode'], samples)
positions = inits[:,:3]
velosities = inits[:,3:]
