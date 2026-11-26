# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
FCP on a two-layer bcc Li(001) slab with an 8x8x1 k-point mesh.

The geometry is specified in Angstrom and converted to Bohr.
Chemical potentials, smearing width and energies are Hartree.
Only equal-weight KRKS in 3D without a barrier is supported.
'''

import numpy as np
import cupy as cp
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf import fcp, slabgate
from gpu4pyscf.pbc.scf.fcp import FCP
from gpu4pyscf.scf import hf

sigma = 0.005  # Ha; broaden the sampled metallic bands.
mu_offset = -0.01  # Ha relative to neutral Li; drive charge depletion.

a = 3.49  # bcc lattice constant in Angstrom.
length_A = 18.0
lattice = np.diag([a, a, length_A]) * 1.889727
# Two adjacent bcc (001) planes, separated by a/2 and centered in the cell.
coords = np.array([
    [0.0, 0.0, length_A / 2 - a / 4],
    [a / 2, a / 2, length_A / 2 + a / 4],
]) * 1.889727

cell = gto.M(
    a=lattice,
    atom=[('Li', coords[0]), ('Li', coords[1])],
    unit='Bohr', dimension=3, charge=0, spin=0,
    basis='gth-dzvp', pseudo='gth-pbe',
    mesh=[51, 51, 541], precision=1e-10, verbose=5)

nelec = float(cell.nelectron)  # 6 explicit electrons: Li GTH-PBE includes 1s.
kpts = cell.make_kpts([8, 8, 1])

hf.overlap_zero_eigenvalue_threshold = 1e-6
mf = dft.KRKS(cell, kpts=kpts, xc='pbe')
mf.grids = dft.UniformGrids(cell)
mf = mf.multigrid_numint()
mf = mf.smearing(sigma=sigma)
mf.conv_tol = 1e-9
mf.conv_tol_grad = 1e-6
mf.max_cycle = 300

length = lattice[2, 2]
slabgate.DIPOLE_WIDTH_RATIO = 0.10

# Obtain the target from the neutral slab before adding the gate.
plain = mf.copy()
plain.kernel()
if not plain.converged:
    raise RuntimeError('Neutral Li SCF did not converge')
mu_neutral = fcp._occupations(plain.mo_energy.get(), nelec, sigma)[0]
mu = mu_neutral + mu_offset
print('plain e_tot:', plain.e_tot)
print('neutral mu:', mu_neutral, 'mu_offset:', mu_offset)
print('mu:', mu)

mf = slabgate.gate_charge(
    mf, zgate=0.70 * length, gate_side=1,
    dipole_gap=0.30 * length, barrier=False)

dm0 = mf.get_init_guess()
overlap = mf.get_ovlp()
dm_nelec = float(cp.einsum('kij,kji->', dm0, overlap).real) / len(kpts)
dm0 *= nelec / dm_nelec

# FCP enables fixed-N Fermi smearing. sigma=None or sigma=0 raises ValueError.
calc = FCP(mf, mu=mu, sigma=sigma)
calc.kernel(nelec=nelec, dm0=dm0)
assert calc.converged

print('N per cell:', calc.nelec)
print('mu, target (Ha):', calc.mu_eff, calc.mu)
print('eta, Lambda (Ha):', calc.eta, calc.shift)
print('Total energy (Ha):', calc.e_tot)
print('Helmholtz free energy (Ha):', calc.e_free)
print('Grand potential (Ha):', calc.e_grand)
