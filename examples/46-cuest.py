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


##############################################################################
#   This example shows the basic usage of GPU4PySCF-cuEST interface.         #
#   Please refer to https://developer.nvidia.com/cuda/cuda-x-libraries/cuest #
#   for cuEST installation and brief explanation, and                        #
#   https://docs.nvidia.com/cuda/cuest/index.html for cuEST detailed usage   #
#   and examples.                                                            #
##############################################################################


# Before you go, please make sure your local CUDA version, GPU4PySCF compilation version, cuEST version, CuPy version,
# are all consistent with each other, i.e. they are all cuda12 or cuda13.


# The basic usage of GPU4PySCF-cuEST interface is to add this line after you construct a PySCF mean-field object:
# mf = apply_cuest_wrapper(mf.to_gpu())
# And that's it. Please always apply the cuest_wrapper at the very end, right before you call kernel().
# Up to version 0.1.0, cuest functionalities are quite limited. Please pay attention to the warning messages
# provided, particularly when trying to get exactly matching results.


import pyscf

from gpu4pyscf.lib.cuest_wrapper import apply_cuest_wrapper

mol = pyscf.M(
    atom = '''
        O       0.0000000000    -0.0000000000     0.1174000000
        H      -0.7570000000    -0.0000000000    -0.4696000000
        H       0.7570000000     0.0000000000    -0.4696000000
    ''',
    basis = 'def2-tzvp',
    charge = 0,
    spin = 0,
    verbose = 4,
)

print("\n\n HF energy and gradient:\n\n")

mf = mol.RHF().density_fit(auxbasis = "def2-universal-jkfit")
mf = apply_cuest_wrapper(mf.to_gpu())
energy = mf.kernel()
assert mf.converged

gobj = mf.Gradients()
gradient = gobj.kernel()

print("\n\n DFT energy and gradient:\n\n")

mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit")
mf.grids.atom_grid = (99,590)
### If you need match results between native GPU4PySCF and cuEST, please apply the following lines as well.
# mf.grids.becke_scheme = stratmann
# mf.grids.radii_adjust = None
# mf.nlcgrids.becke_scheme = stratmann
# mf.nlcgrids.radii_adjust = None

mf = apply_cuest_wrapper(mf.to_gpu())
energy = mf.kernel()
assert mf.converged

gobj = mf.Gradients()
gobj.grid_response = True
gradient = gobj.kernel()

print("\n\n DFT + PCM energy and gradient:\n\n")

mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit")
mf.grids.atom_grid = (99,590)
mf = mf.PCM()
### If you need match results between native GPU4PySCF and cuEST, please apply the following lines as well.
# mf.grids.becke_scheme = stratmann
# mf.grids.radii_adjust = None
# mf.nlcgrids.becke_scheme = stratmann
# mf.nlcgrids.radii_adjust = None
# mf.with_solvent.surface_discretization_method = "ISWIG"

mf = apply_cuest_wrapper(mf.to_gpu())
energy = mf.kernel()
assert mf.converged

gobj = mf.Gradients()
gobj.grid_response = True
gradient = gobj.kernel()

print("\n\n HF energy and gradient with ECP:\n\n")

mol_with_ecp = pyscf.M(
    atom = '''
        H 0.0 0.0 0.0
        I 2.0 0.0 0.0
    ''',
    basis = 'def2-tzvp',
    ecp = 'def2-tzvp',
    charge = 0,
    spin = 0,
    verbose = 4,
)

mf = mol_with_ecp.RHF().density_fit(auxbasis = "def2-universal-jkfit")
mf = apply_cuest_wrapper(mf.to_gpu())
energy = mf.kernel()
assert mf.converged

gobj = mf.Gradients()
gradient = gobj.kernel()

print("\n\n Geometry optimization:\n\n")

mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit")
mf.grids.atom_grid = (99,590)
mf = mf.PCM()

from pyscf.geomopt.geometric_solver import optimize
mol_eq = optimize(mf, maxsteps = 20)
geometry = mol_eq.atom_coords()
print(f"Geometry in Bohr = {repr(geometry)}")

