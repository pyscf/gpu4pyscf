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

'''Hartree-Fock for periodic systems
'''

from .import hf
from . import uhf
from . import khf
from . import kuhf

rhf = hf
krhf = khf

KRHF = krhf.KRHF
KUHF = kuhf.KUHF

def RHF(cell, *args, **kwargs):
    if 'kpts' in kwargs:
        return KRHF(cell, *args, **kwargs)
    if cell.spin == 0:
        return rhf.RHF(cell, *args, **kwargs)
    else:
        raise NotImplementedError

def UHF(cell, *args, **kwargs):
    if 'kpts' in kwargs:
        return KUHF(cell, *args, **kwargs)
    return uhf.UHF(cell, *args, **kwargs)

def HF(cell, *args, **kwargs):
    if cell.spin == 0:
        return RHF(cell, *args, **kwargs)
    else:
        return UHF(cell, *args, **kwargs)
