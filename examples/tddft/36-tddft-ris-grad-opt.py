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
TDDFT-ris excited state gradient and geometry optimization
'''

import pyscf
import gpu4pyscf.tdscf.ris as ris
from gpu4pyscf.dft import rks
from pyscf.geomopt.geometric_solver import optimize

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "ccpvdz"

mol = pyscf.M(
    atom=atom, basis=bas0, max_memory=32000)
mf = rks.RKS(mol, xc='b3lyp').to_gpu()
mf.kernel()
td_ris = ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, GS=True)
td_ris.conv_tol = 1.0E-4
td_ris.Ktrunc = 0.0
td_ris.kernel()

"""
TDDFT-ris excited state geometry optimization
1st usage
"""
mol_gpu = optimize(td_ris)
mff = rks.RKS(mol_gpu, xc='b3lyp').to_gpu()
mff.kernel()
tdf_ris = ris.TDDFT(mf=mff, nstates=5, spectra=False, single=False, GS=True)
tdf_ris.conv_tol = 1.0E-4
tdf_ris.Ktrunc = 0.0
output = tdf_ris.kernel()

"""
TDDFT-ris excited state geometry optimization
2nd usage
"""
excited_grad = td_ris.nuc_grad_method().as_scanner(state=1)
mol_gpu = excited_grad.optimizer().kernel()

"""
TDDFT-ris excited state gradient
"""
excited_gradf_ris = tdf_ris.nuc_grad_method()
excited_gradf_ris.kernel()
