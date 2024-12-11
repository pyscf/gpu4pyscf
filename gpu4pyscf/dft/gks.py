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

from pyscf.dft import gks
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import rks
from gpu4pyscf.scf.ghf import GHF

class GKS(gks.GKS, GHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    def __init__(self, mol, xc='LDA,VWN'):
        raise NotImplementedError

    reset = rks.RKS.reset
    energy_elec = rks.RKS.energy_elec
    get_veff = NotImplemented
    nuc_grad_method = NotImplemented
    to_hf = NotImplemented
