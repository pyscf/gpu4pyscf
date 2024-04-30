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

###########################################
#  Example of unrestricted DFT
###########################################

import pyscf
from gpu4pyscf.dft import uks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

# SCF, gradient, and Hessian for DF-UKS
mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mol.charge = 1
mol.spin = 1
mf = uks.UKS(mol, xc='b3lyp').density_fit()
mf.kernel()

gobj = mf.nuc_grad_method()
g = gobj.kernel()

hobj = mf.Hessian()
h = hobj.kernel()

# SCF, gradient, and Hessian for DF-UKS with IEF-PCM
mf_with_pcm = mf.PCM()
mf_with_pcm.with_solvent.method = 'IEF-PCM'
mf_with_pcm.kernel()

gobj_with_pcm = mf_with_pcm.nuc_grad_method()
g = gobj_with_pcm.kernel()

hobj_with_pcm = mf_with_pcm.Hessian()
h = hobj_with_pcm.kernel()

# SCF, gradient, and Hessian for DF-UKS with IEF-PCM
mf_with_smd = mf.SMD()
mf_with_smd.with_solvent.solvent = 'water'
mf_with_smd.kernel()

gobj_with_smd = mf_with_smd.nuc_grad_method()
g = gobj_with_smd.kernel()

hobj_with_smd = mf_with_smd.Hessian()
h = hobj_with_smd.kernel()
