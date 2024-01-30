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

import numpy as np
import cupy
from pyscf.scf import rohf
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.lib.cupy_helper import tag_array


class ROHF(rohf.ROHF, hf.RHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    get_jk = hf._get_jk
    _eigh = hf.RHF._eigh
    scf = kernel = hf.RHF.kernel
    # FIXME: Needs more tests for get_fock and get_occ
    get_fock = hf.return_cupy_array(rohf.ROHF.get_fock)
    get_occ = hf.return_cupy_array(rohf.ROHF.get_occ)
    get_hcore = hf.RHF.get_hcore
    get_ovlp = hf.RHF.get_ovlp
    get_init_guess = uhf.UHF.get_init_guess
    make_rdm1 = hf.return_cupy_array(rohf.ROHF.make_rdm1)
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
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented
    nuc_grad_method = NotImplemented

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if getattr(dm, 'ndim', 0) == 2:
            dm = cupy.asarray((dm*.5,dm*.5))

        if dm_last is None or not self.direct_scf:
            if getattr(dm, 'mo_coeff', None) is not None:
                mo_coeff = dm.mo_coeff
                mo_occ_a = (dm.mo_occ > 0).astype(np.double)
                mo_occ_b = (dm.mo_occ ==2).astype(np.double)
                dm = tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                               mo_occ=(mo_occ_a,mo_occ_b))
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = cupy.asarray(dm) - cupy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += vhf_last
        return vhf
