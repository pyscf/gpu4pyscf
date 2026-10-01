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

'''Atom-resolved magnetic moments in KUHF/KUKS initial guesses.

Moments are N_alpha - N_beta per primitive cell. cell.spin is the total
over the k-point mesh: matching guesses satisfy nkpts * sum(magmoms) = spin.
The moments seed the density; they are not constraints on the converged SCF.
'''

import cupy as cp
import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import scf
from gpu4pyscf.pbc.tools import get_init_guess_with_magmom


cell = gto.Cell(
    atom='H 3 6 6; H 9 6 6',
    a=np.eye(3) * 12,
    unit='B',
    basis={'H': [[0, (2., 1.)], [1, (2., 1.)]]},
    spin=0,
    verbose=4,
).build()
kpts = cell.make_kpts([2, 1, 1])


def show_guess(label, magmoms, spin, kpts):
    cell.spin = spin
    mf = scf.KUHF(cell, kpts=kpts)
    dm = get_init_guess_with_magmom(cell, kpts, magmoms)
    s = mf.get_ovlp()
    nelec = cp.einsum('skij,kji->s', dm, s).real.get()
    local_moments = [
        float(cp.einsum('kij,kji->', (dm[0] - dm[1])[:, p0:p1],
                        s[:, :, p0:p1]).real) / len(kpts)
        for p0, p1 in cell.aoslice_by_atom()[:, 2:]
    ]
    print(label)
    print('  local moments per cell:', local_moments)
    print('  initial electrons over k-points:', nelec)
    print('  SCF target electrons over k-points:', mf.nelec)
    # Start a calculation with mf.kernel(dm0=dm), or use this density in KUKS.
    return dm


# 1. Integer antiferromagnetic moments: 2 * (1 - 1) = 0.
dm_integer = show_guess('Integer AFM', {0: 1., 1: -1.}, 0, kpts)

# 2. Fractional antiferromagnetic moments, interpolated from atomic densities.
dm_fractional = show_guess('Fractional AFM', {0: .5, 1: -.5}, 0, kpts)

# 3. Fractional moments with a nonzero sum: 2 * (.5 + .5) = cell.spin = 2.
# This is a valid polarized initial guess, not an ill-defined spin state.
dm_polarized = show_guess('Fractional FM', {0: .5, 1: .5}, 2, kpts)

# 4. A mismatch warns and preserves the requested initial moments.
# Neither cell.spin nor the k-points are adjusted. SCF will use mf.nelec.
dm_mismatch = show_guess('Spin mismatch', {0: .5, 1: .5}, 0, kpts)

# Changing the k-point count also changes the required cell.spin.
dm_kmesh_mismatch = show_guess(
    'K-point mesh mismatch', {0: .5, 1: .5}, 2, cell.make_kpts([4, 1, 1]))

# Invalid SCF electron/spin parity is an error, not just a guess mismatch.
cell.spin = 1  # Four electrons over two k-points require an even integer spin.
try:
    get_init_guess_with_magmom(cell, kpts, {0: .5, 1: .5})
except ValueError as err:
    print('Invalid SCF spin:', err)
