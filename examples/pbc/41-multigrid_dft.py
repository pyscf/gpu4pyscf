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
Using multi-grid algorithm for DFT calculation
'''

import numpy as np
import pyscf

cell = pyscf.M(
    a = np.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-dzvp',
    pseudo = 'gth-pbe',
    verbose = 5,
)

#
# Use the multigrid_numint to enable the multigrid integration algorithm.
# This method overwrites the _numint attribute of the DFT object with the
# multigrid integrator.
#
mf = cell.RKS(xc='pbe').to_gpu()
mf = mf.multigrid_numint()
mf.run()
mf.get_bands

kpts = cell.make_kpts([2,2,2])
mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
mf = mf.multigrid_numint()
mf.run()

#
# The multigrid integrator also supports constructing the Fock matrix for
# arbitrary k-points along a band path. This can be used to compute
# band structures. The band path can be generated using the ASE package.
#
import cupy as cp
import matplotlib.pyplot as plt
from pyscf.data.nist import BOHR, HARTREE2EV
from ase.cell import Cell as ase_Cell

a = cell.lattice_vectors() * BOHR # To Angstrom
bp = ase_Cell(a).bandpath(npoints=50)
band_kpts, kpath, sp_points = bp
band_kpts = cell.get_abs_kpts(band_kpts)
e_kn = cp.asnumpy(cp.asarray(mf.get_bands(band_kpts=band_kpts)[0]))

nocc = cell.nelectron // 2
vbmax = max(e_kn[:,nocc-1])
e_kn = (e_kn - vbmax) * HARTREE2EV

def plot_band_structure(bp, e_kn, ax):
    nbands = e_kn.shape[1]
    for b in range(nbands):
        plt.plot(kpath, e_kn[:,b], color='k', linedwidth=1)

    for x in bp.special_points_positions:
        ax.axvline(x, color="gray", linewidth=0.8)

    ax.set_xticks(bp.special_points_positions)
    ax.set_xticklabels(bp.path)

    kdist = bp.distances
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("k-path")
    ax.set_title("Band Structure")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)  # Fermi level
    return ax

fig, ax = plt.subplots(figsize=(6, 4))
plot_band_structure(bp, e_kn, ax)
plt.tight_layout()
plt.show()
