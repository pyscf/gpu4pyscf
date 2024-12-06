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
import cupy as cp
from pyscf.dft import roks as roks_cpu
from gpu4pyscf.scf.rohf import ROHF
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.lib import utils

class ROKS(rks.KohnShamDFT, ROHF):

    def __init__(self, mol, xc='LDA,VWN'):
        ROHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if dm is None:
            dm = self.make_rdm1()
        elif getattr(dm, 'mo_coeff', None) is not None:
            mo_coeff = dm.mo_coeff
            mo_occ_a = (dm.mo_occ > 0).astype(np.double)
            mo_occ_b = (dm.mo_occ ==2).astype(np.double)
            if dm.ndim == 2:
                dm = cp.repeat(dm[None]*.5, 2, axis=0)
            dm = tag_array(dm, mo_coeff=cp.asarray((mo_coeff,mo_coeff)),
                           mo_occ=cp.asarray((mo_occ_a,mo_occ_b)))
        elif dm.ndim == 2:
            dm = cp.repeat(dm[None]*.5, 2, axis=0)
        return uks.get_veff(self, mol, dm, dm_last, vhf_last, hermi)

    energy_elec = uks.UKS.energy_elec
    nuc_grad_method = NotImplemented
    to_hf = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = roks_cpu.ROKS(self.mol, xc=self.xc)
        utils.to_cpu(self, mf)
        return mf
