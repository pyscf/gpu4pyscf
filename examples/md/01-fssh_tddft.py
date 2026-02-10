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
This example demonstrates the FSSH simualtion with TDDFT method
'''

import numpy as np
import pyscf
from gpu4pyscf.md import maxwell_boltzmann_velocities, FSSH_TDDFT
from gpu4pyscf.tdscf.ris import TDA

mol = pyscf.M(
    atom='''
C   -1.302   0.206   0.000
H   -0.768  -0.722   0.000
H   -2.372   0.206   0.000
C   -0.626   1.381   0.000
H   -1.159   2.309   0.000
H    0.444   1.381   0.000
''', basis='6-31g')

mf = mol.RKS(xc='pbe0').to_gpu().density_fit()
td = mf.TDA()
td.nstates = 3

# Initial velocities from a Maxwell–Boltzmann distribution at 300 K.
# Initial positions and velocities can also be generated using the Wigner
# sampling method (see 02-wigner_sampling.py).
v = maxwell_boltzmann_velocities(mol.atom_mass_list(True), temperature=300)

# Run a FSSH simulation including only the first and second excited
# electronic states. This restricts surface hopping among these states.
fssh = FSSH_TDDFT(td, states=[1, 2])
# Set the initial electronic state to the second excited state.
fssh.cur_state = 2
fssh.nsteps = 50 # Number of time steps to propagate.
fssh.time_step = 1.0 # fs
# Save trajectory in an HDF5 file.
fssh.filename = 'c2h4_traj.h5'
# Setting the random seed to generate reproducible trajectory
#fssh.seed = 9592
# coefficient for each electronic state, corresponding to the initial state
# which is solely contributed by the second electronic excited state.
coefficient = np.array([0.0,1.0])
fssh.kernel(velocity=v, coefficient=coefficient)

# Convert the HDF5 trajectory file to a human-readable XYZ format.
from gpu4pyscf.md.fssh import h5_to_xyz
h5_to_xyz(fssh.filename, 'traj_50.xyz')

#
# Restart or continue a FSSH simulation from a previous trajectory
#
fssh = FSSH_TDDFT(td, states=[1,2])
# Restore the FSSH from an existing HDF5 trajectory file. Attributes of the FSSH
# instance are updated by this operation.
fssh.restore('c2h4_traj.h5')
fssh.nsteps = 100
# kernel will read the position, velocity and coefficient from the last step of
# the previous run.
fssh.kernel()

h5_to_xyz(fssh.filename, 'traj_100.xyz')
