# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

import cupy
from pyscf.scf import ghf
from gpu4pyscf.scf.hf import _get_jk as _get_jk_nr
from gpu4pyscf.scf.hf import eigh

class GHF(ghf.GHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        nao = mol.nao
        dm = cupy.asarray(dm)

        def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
            return _get_jk_nr(self, mol, dm, hermi, with_j, with_k, omega)

        if nao == dm.shape[-1]:
            vj, vk = jkbuild(mol, dm, hermi, with_j, with_k, omega)
        else:  # GHF density matrix, shape (2N,2N)
            vj, vk = ghf.get_jk(mol, dm, hermi, with_j, with_k, jkbuild, omega)
        return vj, vk

    _eigh = staticmethod(eigh)
