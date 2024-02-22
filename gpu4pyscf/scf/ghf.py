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
from gpu4pyscf.scf import hf

class GHF(ghf.GHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    _eigh = hf.RHF._eigh
    scf = kernel = hf.RHF.kernel
    get_hcore = hf.return_cupy_array(ghf.GHF.get_hcore)
    get_ovlp = hf.return_cupy_array(ghf.GHF.get_ovlp)
    get_init_guess = hf.RHF.get_init_guess
    make_rdm2 = NotImplemented
    dump_chk = NotImplemented
    newton = NotImplemented
    x2c = x2c1e = sfx2c1e = NotImplemented
    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented
    to_ks = NotImplemented
    canonicalize = NotImplemented
    # TODO: Enable followings after testing
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        nao = mol.nao
        dm = cupy.asarray(dm)

        def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
            return hf._get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        if nao == dm.shape[-1]:
            vj, vk = jkbuild(mol, dm, hermi, with_j, with_k, omega)
        else:  # GHF density matrix, shape (2N,2N)
            vj, vk = ghf.get_jk(mol, dm, hermi, with_j, with_k, jkbuild, omega)
        return vj, vk
