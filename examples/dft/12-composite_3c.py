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
#  Composite 3c methods (r2SCAN-3c, B97-3c) through the xc name
#
#  The xc name selects the functional, its dispersion correction and its
#  gCP/SRB basis-set correction, the same as in PySCF (>= 2.15). The
#  basis is not implied: r2SCAN-3c is defined with def2-mTZVPP, B97-3c with
#  def2-mTZVP. All three corrections enter the energy, gradient and Hessian.
###########################################################

import pyscf
from gpu4pyscf.dft import rks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-mtzvpp')
mf = rks.RKS(mol, xc='r2scan-3c').density_fit()
e_tot = mf.kernel()
print('r2SCAN-3c energy', e_tot)
print('D4 dispersion', mf.scf_summary['dispersion'])
print('gCP correction', mf.scf_summary['gcp'])

g = mf.Gradients().kernel()
h = mf.Hessian().kernel()

# The gCP term can be switched off, e.g. to compare with a code that lacks it
mf_no_gcp = rks.RKS(mol, xc='r2scan-3c').density_fit()
mf_no_gcp.gcp = False
print('energy without gCP', mf_no_gcp.kernel())
