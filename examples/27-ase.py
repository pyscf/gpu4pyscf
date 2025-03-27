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

###################################
#  Example of ASE interface
###################################

from ase import Atoms
from gpu4pyscf.tools import get_default_config
from gpu4pyscf.tools.ase_interface import PySCFCalculator

atoms = Atoms('H2O', positions=[(0.76, 0.58, 0.0),
                               (-0.76, 0.58, 0.0),
                               (0.0, 0.0, 0.0)])

# Default method: b3lyp/def2-tzvpp, DF, (99,590)
config = get_default_config()     
calc = PySCFCalculator(config)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

## 
## Geometry optimization with LBFGS
##    pip3 install ase
## See more functionalities in https://wiki.fysik.dtu.dk/ase/
##
from ase.optimize import LBFGS
dyn = LBFGS(atoms, logfile='opt.log')
dyn.run(fmax=0.02)
print("Final energy (Hartree):", atoms.get_potential_energy())
print("Final geometry (Angstrom):")
print(atoms.get_positions())

## 
## Transition State Search with Sella
##    (pip3 install sella)
##  See details in https://github.com/zadorlab/sella/wiki
##

from sella import Sella
positions = [
    [  0.000,  0.000,  0.000],  # C
    [  0.630,  0.630,  0.630],  # H
    [  0.630, -0.630, -0.630],  # H
    [ -0.630,  0.630, -0.630],  # H
    [  0.000,  0.000,  2.50 ],  # Br (slightly away from C)
    [  0.000,  0.000, -2.80]    # Cl (approaching from opposite side)
]
symbols = ['C', 'H', 'H', 'H', 'Br', 'Cl']
atoms = Atoms(symbols=symbols, positions=positions)
config = get_default_config()
config['charge'] = -1
config['verbose'] = 0
calc = PySCFCalculator(config)
atoms.set_calculator(calc)
opt = Sella(
    atoms,
    internal=True,
)
opt.run(fmax=0.05, steps=300)
