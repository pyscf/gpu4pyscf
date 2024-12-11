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
