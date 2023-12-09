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
import time
from pyscf import lib

from gpu4pyscf.dft import rks
lib.num_threads(8)

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

start_time = time.time()
mol = pyscf.M(
    atom='Vitamin_C.xyz',
    verbose=4)
# set verbose >= 6 for debugging timer

print(f'{mol.nao} atomic orbitals')
mf = rks.RKS(mol, xc='HYB_MGGA_XC_WB97M_V').density_fit()
mf.grids.level = 5
mf.conv_tol = 1e-8
mf.direct_scf_tol = 1e-14
mf.nlcgrids.level = 2
e_tot = mf.kernel()
end_time = time.time()
print(f'Wallclock time: {end_time-start_time}')
