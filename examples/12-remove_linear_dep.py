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
from pyscf import lib, gto

from gpu4pyscf.dft import rks
lib.num_threads(8)

mol = gto.M(atom=['H 0 0 %f'%i for i in range(10)], unit='Bohr',
            basis='ccpvtz', symmetry=1)
mol.verbose = 4

mf = rks.RKS(mol, xc='B3LYP').density_fit()
mf.grids.level = 5
mf.conv_tol = 1e-10
mf.direct_scf_tol = 1e-14
mf.nlcgrids.level = 2

import cupy
from functools import reduce
tol = 1e-6
def eig(h, s):
    d, t = cupy.linalg.eigh(s)
    x = t[:,d>tol] / cupy.sqrt(d[d>tol])
    xhx = reduce(cupy.dot, (x.T, h, x))
    e, c = cupy.linalg.eigh(xhx)
    c = cupy.dot(x, c)
    return e, c
from gpu4pyscf import scf
scf.hf.eigh = eig

e_tot = mf.kernel()
end_time = time.time()
