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

import numpy as np
import pyscf
from gpu4pyscf.scf import RHF
from gpu4pyscf.mp import dfmp2

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas = 'ccpvdz'

mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
mf = RHF(mol).density_fit()
e_hf = mf.kernel()

ptobj = dfmp2.DFMP2(mf)
e_corr, t2 = ptobj.kernel()
e_mp2 = e_hf + e_corr

# frozen core
ptobj.frozen = [0]
e_corr, t2 = ptobj.kernel()
e_mp2 = e_hf + e_corr