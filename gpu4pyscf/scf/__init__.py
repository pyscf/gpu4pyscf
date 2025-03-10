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

from . import hf
from .uhf import UHF
from .ghf import GHF
from .rohf import ROHF
from . import dispersion

def HF(mol, *args):
    if mol.nelectron == 1 or mol.spin == 0:
        return RHF(mol, *args)
    else:
        return UHF(mol, *args)

def RHF(mol, *args):
    from gpu4pyscf.lib.cupy_helper import get_avail_mem
    from . import hf_lowmem
    mem = get_avail_mem()
    nao = mol.nao
    if nao**2*30*8 > mem:
        return hf_lowmem.RHF(mol, *args)
    else:
        return hf.RHF(mol, *args)
