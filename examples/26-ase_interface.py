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

####################################################
# Example of using ASE with GPU4PySCF
####################################################


from ase.calculators.calculator import Calculator, all_changes
from pyscf import gto, scf, grad

class PySCFCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, basis='sto-3g', xc='PBE', charge=0, spin=0):
        super().__init__()
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.pyscf_mol = None
        self.calc = None

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Extract atomic symbols and positions
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        # Create PySCF molecule
        self.pyscf_mol = gto.M(
            atom=[[symbols[i], positions[i]] for i in range(len(symbols))],
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
        )

        # Set up PySCF calculation
        self.calc = scf.RKS(self.pyscf_mol)
        self.calc.xc = self.xc
        energy = self.calc.kernel()

        # Store energy result
        self.results['energy'] = energy

        # Compute forces using PySCF gradients
        mf_grad = grad.RKS(self.calc)
        forces = -mf_grad.kernel()
        self.results['forces'] = forces


from ase import Atoms
from sella import Sella

# Define a water molecule
molecule = Atoms('H2O', 
                 positions=[
                     [0.0, 0.0, 0.0], 
                     [0.0, 0.0, 1.0], 
                     [1.0, 0.0, 0.0]])

# Attach the custom PySCF calculator
molecule.calc = PySCFCalculator(basis='sto-3g', xc='PBE')

# Optimize geometry using Sella
optimizer = Sella(molecule)
optimizer.run(fmax=0.05)  # Converge when forces are below 0.05 eV/Å

# Print results
print("Optimized positions (Å):")
print(molecule.get_positions())
print("Final energy (eV):", molecule.get_potential_energy())
