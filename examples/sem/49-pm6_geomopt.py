# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

###########################################################
#  Example of PM6 geometry optimization
###########################################################

from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf import sem

atoms = """
    O 0.0000 0.0000 0.0000
    H 0.7570 0.5860 0.0000
    H -0.7570 0.5860 0.0000
    """

# Create a PM6 molecule object
mol = sem.gto.mole.Mole(atoms, verbose=4)
mol.build()

# Create a PM6 mean-field object and run an initial SCF
mf = sem.scf.hf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# Optimize the geometry with the geomeTRIC solver. The driver internally calls
# mf.nuc_grad_method().as_scanner() to evaluate energies and gradients along the
# optimization path.
mol_eq = optimize(mf, maxsteps=50)

print("Optimized coordinates (Angstrom):")
print(mol_eq.atom_coords(unit='Angstrom'))

# Recompute the energy at the optimized geometry
mf_eq = sem.scf.hf.RHF(mol_eq)
mf_eq.conv_tol = 1e-12
e_opt = mf_eq.kernel()
print("PM6 energy at the optimized geometry (Hartree): %.10f" % e_opt)
