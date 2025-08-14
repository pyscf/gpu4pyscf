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

try:
    from ase.calculators.calculator import Calculator, all_properties
except ImportError:
    print("""ASE is not found. Please install ASE via
pip3 install ase
          """)
    raise RuntimeError("ASE is not found")

import numpy as np
from ase.units import Debye
from pyscf import lib
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscf.gto.mole import charge
from pyscf.pbc.gto.cell import Cell
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

# These functions are copied from the development branch of PySCF and will be
# provided by the pyscf.pbc.tools.pyscf_ase module in PySCF 2.11.

def cell_from_ase(ase_atoms):
    '''Convert ASE atoms to PySCF Cell instance. The lattice vectors and atomic
    positions are defined in the Cell instance. It does not have any basis sets
    or pseudopotentials assigned. The Cell instance is not initialized with 'build()'.
    '''
    cell = Cell()
    cell.atom = ase_atoms_to_pyscf(ase_atoms)
    cell.a = np.asarray(ase_atoms.cell)
    return cell

class PySCF(Calculator):
    implemented_properties = ['energy', 'forces', 'stress',
                              'dipole', 'magmom']

    default_parameters = {}

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='PySCF', atoms=None, directory='.', method=None,
                 **kwargs):
        """Construct PySCF-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'PySCF'.

        method: A PySCF method class
        """
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, directory=directory, **kwargs)

        if not isinstance(method, lib.StreamObject):
            raise RuntimeError(f'{method} must be an instance of a PySCF method')

        self.method = method
        self.pbc = hasattr(method, 'cell')
        if self.pbc:
            mol = method.cell
        else:
            mol = method.mol
        if not mol.unit.startswith(('A','a')):
            raise RuntimeError("PySCF unit must be A to work with ASE")
        self.mol = mol
        self.method_scan = None
        if hasattr(method, 'as_scanner'):
            # Scanner can utilize the initial guess from previous calculations
            self.method_scan = method.as_scanner()

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_properties):
        Calculator.calculate(self, atoms)

        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        Z = np.array([charge(x) for x in self.mol.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))

        if self.pbc:
            self.mol.set_geom_(_atoms, a=atoms.cell)
        else:
            self.mol.set_geom_(_atoms)

        with_grad = 'forces' in properties or 'stress' in properties
        with_energy = with_grad or 'energy' in properties or 'dipole' in properties

        if with_energy:
            if self.method_scan is None:
                self.mol.set_geom_(atoms)
                self.method.reset(self.mol).run()
                e_tot = self.method.e_tot
                if not getattr(self.method, 'converged', True):
                    raise RuntimeError(f'{self.method} not converged')
            else:
                e_tot = self.method_scan(self.mol)
                if not self.method_scan.converged:
                    raise RuntimeError(f'{self.method} not converged')
            self.results['energy'] = e_tot * HARTREE2EV

        if self.method_scan is None:
            base_method = self.method
        else:
            base_method = self.method_scan

        if with_grad:
            grad_obj = base_method.Gradients()

        if 'forces' in properties:
            forces = -grad_obj.kernel()
            self.results['forces'] = forces * (HARTREE2EV / BOHR)

        if 'stress' in properties:
            stress = grad_obj.get_stress()
            self.results['stress'] = stress * (HARTREE2EV / BOHR)

        if 'dipole' in properties:
            if self.pbc:
                raise NotImplementedError('dipole for PBC calculations')
            # in Gaussian cgs unit
            self.results['dipole'] = base_method.dip_moment() * Debye

        if 'magmom' in properties:
            magmom = self.mol.spin
            self.results['magmom'] = magmom
