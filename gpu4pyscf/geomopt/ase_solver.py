#!/usr/bin/env python
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

'''
Interface to ASE for lattice and atom position optimization
https://ase-lib.org/ase/optimize.html
'''

import numpy as np
from ase.optimize import BFGS
from ase.filters import UnitCellFilter, StrainFilter
from pyscf import lib
from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms
from gpu4pyscf.tools.ase_interface import PySCF

def kernel(method, target='cell', logfile=None, fmax=0.05, max_steps=100,
           restart=False):
    '''Optimize geometry using ASE.

    Kwargs:
        target : string
            'cell': Optimize both lattice and atom positions within the cell.
            'lattice': Optimize lattice while while fixing the scaled atom positions.
            'atoms': Optimize atom positions only
        logfile: file object, Path, or str
            File to save the ASE output

    Addtional kwargs for ASE optimizer:
        fmax : float
            Convergence criterion of the forces (unit eV/A^3) on atoms.
        max_steps : int
            Number of optimizer steps to be run.
        restart : bool
            Whether to restart from a previus optimization.
    '''
    assert not restart
    cell = method.cell
    atoms = pyscf_to_ase_atoms(cell)
    atoms.calc = ase_calculator = PySCF(method=method)

    if target == 'cell':
        atoms = UnitCellFilter(atoms)
    elif target == 'lattice':
        atoms = StrainFilter(atoms)

    if logfile is None:
        logfile = '-' # stdout

    opt = BFGS(atoms, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=max_steps)

    if target == 'cell' or target == 'lattice':
        atoms = atoms.atoms
    cell = cell.set_geom_(atoms.get_positions(), unit='Ang', a=atoms.cell, inplace=False)
    return converged, cell

class GeometryOptimizer(lib.StreamObject):
    '''Optimize the atom positions and lattice for the input method.

    Attributes:
        fmax : float
            Convergence criterion of the forces on atoms.
        max_steps : int
            Number of optimizer steps to be run.
        target : string
            'cell': Optimize both lattice and atom positions within the cell.
            'lattice': Optimize lattice while while fixing the scaled atom positions.
            'atoms': Optimize atom positions only
        logfile: file object, Path, or str
            File to save the ASE output


    Saved results:
        converged : bool
            Whether the geometry optimization is converged

    Note method.cell will be modified after calling the .kernel() method.
    '''
    def __init__(self, method):
        self.method = method
        self.converged = False
        self.max_steps = 100
        self.fmax = 0.05
        self.target = 'cell'
        self.logfile = None

    @property
    def max_cycle(self):
        return self.max_steps

    @property
    def cell(self):
        return self.method.cell

    @cell.setter
    def cell(self, x):
        assert hasattr(self.method, 'cell')
        self.method.cell = x

    def kernel(self):
        self.converged, self.cell = kernel(
            self.method, self.target, self.logfile,
            fmax=self.fmax, max_steps=self.max_steps)
        return self.cell
    optimize = kernel
