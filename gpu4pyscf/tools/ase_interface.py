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
    from ase import Atoms
    from ase import units
except ImportError:
    print("""ASE is not found. Please install ASE via
pip3 install ase
          """)
    raise RuntimeError("ASE is not found")
import numpy as np
import copy
from pyscf import gto
from pyscf.lib import logger
from pyscf.data.nist import BOHR, HARTREE2EV
from gpu4pyscf import scf, dft

from gpu4pyscf.tools import method_from_config

class PySCFCalculator(Calculator):
    """
    An ASE Calculator that uses GPU4PySCF for quantum chemistry calculations.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, pyscf_config, 
                 **kwargs):

        super().__init__(**kwargs)
        self.pyscf_config = copy.deepcopy(pyscf_config)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_properties):
        """
        The main interface with ASE. This method is called automatically when
        ASE requires energies/forces, etc.
        """
        # The Calculator base class requires calling this for bookkeeping
        Calculator.calculate(self, atoms, properties, system_changes)

        # Extract geometry and atomic numbers from ASE
        positions = atoms.get_positions()  # in self.unit
        atomic_numbers = atoms.get_atomic_numbers()
        atom = [(Z, tuple(pos)) for Z, pos in zip(atomic_numbers, positions)]

        # Build the PySCF object
        self.pyscf_config['atom'] = atom
        self.pyscf_config['logfile'] = None
        mf = method_from_config(self.pyscf_config)

        # Run the SCF
        mf.run()
        if not mf.converged:
            logger.error(mf, 'SCF failed to converge')

        # Compute total energy
        energy = mf.e_tot * HARTREE2EV
        
        gcalc = mf.nuc_grad_method()

        grad_mat = gcalc.kernel()  # shape (natm, 3)
        forces = -grad_mat * (HARTREE2EV / BOHR)

        # Store results
        self.results = {
            'energy': energy,
            'forces': forces
        }
    