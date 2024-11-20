# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


##############################################################################
#   This example shows the basic usage of PySCF/GPU4PySCF                    #
#   For the complete guide, please refer to                                  #
#     - PySCF user guide (https://pyscf.org/user.html)                       #
#     - PySCF examples (https://github.com/pyscf/pyscf/tree/master/examples) #
##############################################################################

import numpy as np
import pyscf
from pyscf.hessian import thermo
from gpu4pyscf.dft import rks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(
    atom=atom,                         # water molecule
    basis='def2-tzvpp',                # basis set
    output='./pyscf.log',              # save log file
    verbose=6                          # control the level of print info
    )

mf_GPU = rks.RKS(                      # restricted Kohn-Sham DFT
    mol,                               # pyscf.gto.object
    xc='b3lyp'                         # xc funtionals, such as pbe0, wb97m-v, tpss,
    ).density_fit()                    # density fitting

mf_GPU.grids.atom_grid = (99,590)      # (99,590) lebedev grids, (75,302) is often enough
mf_GPU.conv_tol = 1e-10                # controls SCF convergence tolerance
mf_GPU.max_cycle = 50                  # controls max iterations of SCF
mf_GPU.conv_tol_cpscf = 1e-3           # controls max iterations of CPSCF (for hessian)

# Compute Energy
e_dft = mf_GPU.kernel()
print(f"total energy = {e_dft}") # -76.26736519501688

# Compute Gradient
g = mf_GPU.nuc_grad_method()
g_dft = g.kernel()

# Compute Hessian
h = mf_GPU.Hessian()
h.auxbasis_response = 2                # 0: no aux contribution, 1: some contributions, 2: all
mf_GPU.cphf_grids.atom_grid = (50,194) # customize grids for solving CPSCF equation, SG1 by default
h_dft = h.kernel()

# harmonic analysis
results = thermo.harmonic_analysis(mol, h_dft)
thermo.dump_normal_mode(mol, results)

results = thermo.thermo(
    mf_GPU,                            # GPU4PySCF object
    results['freq_au'],
    298.15,                            # room temperature
    101325)                            # standard atmosphere

thermo.dump_thermo(mol, results)

# force translational symmetry
natm = mol.natm
h_dft = h_dft.transpose([0,2,1,3]).reshape(3*natm,3*natm)
h_diag = h_dft.sum(axis=0)
h_dft -= np.diag(h_diag)
