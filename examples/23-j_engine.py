# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

'''
Compute J and K matrices separately. The J matrix is evaluated using J-engine.
'''

import pyscf
from gpu4pyscf import scf
from gpu4pyscf.scf import jk

mol = pyscf.M(
atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
basis='def2-tzvp',
verbose=5
)

def get_veff(self, mol, dm, *args, **kwargs):
    vj = jk.get_j(mol, dm[0] + dm[1], hermi=1)
    _, vk = jk.get_jk(mol, dm, hermi=1, with_j=False)
    return vj - vk

scf.uhf.UHF.get_veff = get_veff

mf = mol.UHF().to_gpu().run()
