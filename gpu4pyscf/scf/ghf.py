# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
