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

from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm import chelpg

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    [1 , (1. ,  0.     , 0.000)],
    [1 , (0. ,  1.     , 0.000)],
    [1 , (0. , -1.517  , 1.177)],
    [1 , (0. ,  1.517  , 1.177)] ]
mol.basis = '631g'
mol.unit = 'B'
mol.build()
mol.verbose = 6

xc = 'b3lyp'
mf = rks.RKS(mol, xc=xc)
mf.grids.level = 5
mf.kernel()
q = chelpg.eval_chelpg_layer_gpu(mf)
print('partial charge with CHELPG')
print(q) # [ 0.04402311  0.11333945 -0.25767919  0.10031663]