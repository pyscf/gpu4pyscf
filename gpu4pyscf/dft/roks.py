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
from pyscf.dft import roks
from gpu4pyscf.dft import numint
from gpu4pyscf.scf.rohf import ROHF
from gpu4pyscf.dft import uks, gen_grid
from gpu4pyscf.lib.cupy_helper import tag_array

class ROKS(roks.ROKS, ROHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    def __init__(self, mol, xc='LDA,VWN'):
        super().__init__(mol, xc)
        self._numint = numint.NumInt()
        self.disp = None
        self.screen_tol = 1e-14

        grids_level = self.grids.level
        self.grids = gen_grid.Grids(mol)
        self.grids.level = grids_level

        nlcgrids_level = self.nlcgrids.level
        self.nlcgrids = gen_grid.Grids(mol)
        self.nlcgrids.level = nlcgrids_level

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if getattr(dm, 'mo_coeff', None) is not None:
            mo_coeff = dm.mo_coeff
            mo_occ_a = (dm.mo_occ > 0).astype(np.double)
            mo_occ_b = (dm.mo_occ ==2).astype(np.double)
            dm = tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                           mo_occ=(mo_occ_a,mo_occ_b))
        return uks.get_veff(self, mol, dm, dm_last, vhf_last, hermi)

    energy_elec = uks.UKS.energy_elec
    nuc_grad_method = NotImplemented
    to_hf = NotImplemented
