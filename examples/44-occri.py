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

"""
Using multi-grid algorithm for DFT calculation
"""

import numpy as np
import pyscf


from pyscf.pbc import gto

cell = gto.Cell(
    a=np.eye(3) * 3.5668,
    atom="""C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751""",
    basis='gth-dzvp',
    pseudo='gth-hf-rev',
    verbose=5,
    exp_to_discard=0.1,
    ke_cutoff=60,
)

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

from gpu4pyscf.pbc.scf import KRHF, KUHF
from gpu4pyscf.pbc.df.fft import FFTDF, OccRI

rhf = KRHF(cell, kpts)

rhf.with_df = OccRI(cell, kpts)
rhf.exxdiv = 'ewald'
rhf.conv_tol = 1e-6
rhf.max_cycle = 50
e_tot = rhf.kernel()
print('KRHF energy with OccRI: %12.8f' % e_tot)

dm0 = rhf.make_rdm1()

rhf.with_df = FFTDF(cell, kpts)
rhf.exxdiv = 'ewald'
rhf.conv_tol = 1e-6
rhf.max_cycle = 50
e_tot = rhf.kernel()
print('KRHF energy with FFTDF: %12.8f' % e_tot)

uhf = KUHF(cell, kpts)
uhf.with_df = OccRI(cell, kpts)
uhf.exxdiv = 'ewald'
uhf.conv_tol = 1e-6
uhf.max_cycle = 50
e_tot = uhf.kernel()
print('KUHF energy with OccRI: %12.8f' % e_tot)

dm0 = uhf.make_rdm1()

uhf.with_df = FFTDF(cell, kpts)
uhf.exxdiv = 'ewald'
uhf.conv_tol = 1e-6
uhf.max_cycle = 50
e_tot = uhf.kernel()
print('KUHF energy with FFTDF: %12.8f' % e_tot)
