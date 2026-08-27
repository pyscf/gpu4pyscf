#!/usr/bin/env python
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

import numpy as np
from pyscf import lib
from pyscf.pbc import gto

from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.hessian import GammaHessian


a = 5.431
cell = gto.Cell()
cell.a = np.array(
    [
        [0.0, a / 2, a / 2],
        [a / 2, 0.0, a / 2],
        [a / 2, a / 2, 0.0],
    ]
)
cell.atom = [
    ["Si", [0.0, 0.0, 0.0]],
    ["Si", [a / 4, a / 4, a / 4]],
]
cell.unit = "Angstrom"
cell.basis = "gth-dzv"
cell.pseudo = "gth-pbe"
cell.precision = 1e-8
cell.verbose = 4
cell.build()

mf = dft.RKS(cell, xc="pbe")
mf.conv_tol = 1e-9
mf.kernel()

hessian = GammaHessian(mf, primitive_matrix=np.eye(3))
fc = hessian.kernel()
frequencies, eigenvectors, dyn_mat = hessian.phonon_modes()
frequencies_phonopy, _, _ = hessian.phonopy_modes()

lib.logger.note(mf, "Gamma-point frequencies (cm^-1):")
for mode, (frequency, frequency_phonopy) in enumerate(
    zip(frequencies, frequencies_phonopy), 1):
    lib.logger.note(
        mf,
        "  mode %2d: manual %12.4f  phonopy %12.4f",
        mode,
        frequency,
        frequency_phonopy,
    )
