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

from ase.build import bulk
from gpu4pyscf.tools.ase_interface import PySCF, cell_from_ase

atoms = bulk('Si', 'diamond', a=5.4)
cell = cell_from_ase(atoms)
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pbe'
cell.verbose = 4
mf = cell.KRKS(xc='pbe', kpts=cell.make_kpts([3,3,3])).to_gpu()
atoms.calc = PySCF(method=mf)

# 
# Atom position relaxation
#
from ase.optimize import BFGS
from ase.filters import UnitCellFilter, StrainFilter
opt = BFGS(atoms, logfile='atom_opt.log')
opt.run()
print(atoms.get_positions())

#
# Optimize lattice only. Atom positions (fractional coordinates) are frozen.
#
opt = BFGS(StrainFilter(atoms), logfile='lattice_opt.log')
opt.run()
print(atoms.cell)

#
# Optimize both lattice and atom positions
#
opt = BFGS(UnitCellFilter(atoms), logfile='lattice_atom_opt.log')
opt.run()
print(atoms.get_positions())
print(atoms.cell)
