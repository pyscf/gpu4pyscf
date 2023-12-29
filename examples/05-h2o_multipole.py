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

import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp', max_memory=32000)
mol.verbose = 1
mf = rks.RKS(mol, xc='B3LYP').density_fit(auxbasis='def2-tzvpp-jkfit')
mf.kernel()
dm = mf.make_rdm1()

dip = mf.dip_moment(unit='DEBYE', dm=dm.get())
print('dipole moment:')
print(dip)

quad = mf.quad_moment(unit='DEBYE-ANG', dm=dm.get())
print('quadrupole moment:')
print(quad)